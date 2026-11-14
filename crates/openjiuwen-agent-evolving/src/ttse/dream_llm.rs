//! Model-backed dreaming: LLM cluster merges and the tip purge pass.
//!
//! Ports the LLM half of Python `openjiuwen.agent_evolving.ttse.dream`: clusters
//! of near-duplicate tips are consolidated by a model that answers `THINKING /
//! REASON / VERDICT / CANONICAL / MERGE_INDICES / KEEP_INDICES`, and a second
//! prompt quality-checks tip form, answering one
//! `INDEX: i | VERDICT: KEEP|PURGE | REASON: ...` line per tip.
//!
//! Everything the model says is advisory: an unparseable answer leaves the
//! cluster distinct, a blank or unusable `CANONICAL` is rejected, and clusters
//! are always built inside one [`TipCategory`], so a merge can never fold rules
//! from different categories together. The deterministic
//! [`Dreamer::dream`](crate::ttse::Dreamer::dream) pass remains the fallback and
//! the no-model path of [`Dreamer::dream_with_model`].
//!
//! The prompts and the answer parsers live in `dream_prompt`; this module owns
//! the passes that drive a model.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_core::foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_core::foundation::prompt::BaseMessage;

use crate::ttse::bank::{jaccard, tokenize, TipBank};
use crate::ttse::dream::{connected_clusters, find, merge_threshold, Dreamer, MIN_CLUSTER_SIZE};
use crate::ttse::dream_prompt::{
    form_defect, merge_prompt, normalize_merge_subset, parse_merge_verdict, parse_purge_verdicts,
    purge_prompt, MergeDecision, MergeVerdict, PurgeDecision, PurgeVerdict,
    DREAM_MERGE_INSTRUCTION, DREAM_PURGE_INSTRUCTION, MAX_PROMPT_RULES, PURGE_BATCH_SIZE,
    PURGE_MAX_CHARS,
};
use crate::ttse::index::{cosine, EmbeddingIndex};
use crate::ttse::models::{tip_id_for, DreamReport, TipCategory, TipRecord};
use crate::ttse::parse::TipParser;

impl Dreamer {
    /// Installs the model driving the merge and purge passes.
    ///
    /// Model errors and unparseable answers never fail a pass: the affected
    /// cluster stays distinct and a tip stays until a later pass. The
    /// deterministic [`Dreamer::dream`] path is unaffected.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>, request: ModelRequestConfig) -> Self {
        self.model = Some(model);
        self.request = request;
        self
    }

    /// Whether a model is installed, so [`Dreamer::dream_with_model`] can merge
    /// and purge through it.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Overrides how many merge prompts one pass sends.
    #[must_use]
    pub fn with_max_llm_merges(mut self, max_llm_merges: usize) -> Self {
        self.max_llm_merges = max_llm_merges;
        self
    }

    /// How many merge prompts one pass sends at most.
    #[must_use]
    pub fn max_llm_merges(&self) -> usize {
        self.max_llm_merges
    }

    /// Runs one dream pass, using the model for merging and purging when one is
    /// installed.
    ///
    /// Without a model this is exactly [`Dreamer::dream`]. With one the pass has
    /// three stages, in Python's order:
    ///
    /// 1. **prune** — [`TipBank::prune`] applies the configured TTL when
    ///    [`TtseConfig::dream_prune_enabled`](crate::ttse::TtseConfig) is set.
    /// 2. **merge** — clusters are built inside one [`TipCategory`] by cosine
    ///    similarity over `index` when vectors are available (union-find,
    ///    `>= dream_merge_threshold`, `>= dream_cluster_min_size`) and by token
    ///    Jaccard otherwise. Clusters are processed largest first; at most
    ///    [`Dreamer::max_llm_merges`] prompts are sent. A `MERGE`/`REWRITE`
    ///    verdict is applied through `TipBank::merge_cluster`, and its
    ///    `CANONICAL` text replaces the representative's rule.
    /// 3. **purge** — one prompt checks the tips whose form has not been checked
    ///    yet; `PURGE` removes a tip and `KEEP` marks it checked. Rules that are
    ///    blank, too short or rejected by the parser are purged even when the
    ///    model keeps them, which is the deterministic fallback.
    ///
    /// # Errors
    ///
    /// Currently infallible for well-formed banks; the [`Result`] keeps the
    /// signature stable for callers that later need to surface storage errors.
    pub async fn dream_with_model(
        &mut self,
        bank: &mut TipBank,
        index: Option<&EmbeddingIndex>,
    ) -> Result<DreamReport> {
        let Some(model) = self.model.clone() else {
            return Ok(self.dream(bank));
        };
        let examined = bank.len();
        let mut report = DreamReport {
            examined,
            ..DreamReport::default()
        };
        if self.config().dream_prune_enabled {
            report.pruned += bank.prune(self.config());
        }

        let clusters = self.select_clusters(bank, index);
        let min_size = self.config().dream_cluster_min_size.max(MIN_CLUSTER_SIZE);
        let mut budget = self.max_llm_merges;
        for cluster in clusters {
            if budget == 0 {
                break;
            }
            let work = capped_cluster(bank, &cluster, MAX_PROMPT_RULES);
            if work.len() < min_size {
                continue;
            }
            budget -= 1;
            let similarities = pairwise_similarities(bank, index, &work);
            let prompt = merge_prompt(bank, &work, &similarities);
            let Some(answer) = invoke(&model, &self.request, DREAM_MERGE_INSTRUCTION, prompt).await
            else {
                continue;
            };
            let Some(verdict) = parse_merge_verdict(&answer, work.len()) else {
                tracing::debug!(
                    model = model.model_name(),
                    "ttse dream merge answer is unparseable, keeping the cluster distinct"
                );
                continue;
            };
            if let Some(absorbed) = apply_merge(bank, &work, &verdict) {
                report.merged += absorbed;
                report.promoted += 1;
            }
        }

        report.pruned += self.purge(bank).await?;
        Ok(report)
    }

    /// Builds the per-category clusters the pass merges.
    fn select_clusters(&self, bank: &TipBank, index: Option<&EmbeddingIndex>) -> Vec<Vec<String>> {
        let min_size = self.config().dream_cluster_min_size.max(MIN_CLUSTER_SIZE);
        let threshold = merge_threshold(self.config().dream_merge_threshold);
        let mut by_category: BTreeMap<&'static str, Vec<&TipRecord>> = BTreeMap::new();
        for tip in bank.tips() {
            by_category
                .entry(tip.category.as_str())
                .or_default()
                .push(tip);
        }
        let mut clusters: Vec<Vec<String>> = Vec::new();
        for tips in by_category.values() {
            let mut ids: Vec<String> = tips.iter().map(|tip| tip.tip_id.clone()).collect();
            ids.sort();
            let mut found = if let Some(index) = index {
                embedding_clusters(index, &ids, threshold)
            } else {
                let token_sets: Vec<Vec<String>> = ids
                    .iter()
                    .map(|tip_id| {
                        bank.get(tip_id)
                            .map_or_else(Vec::new, |tip| tokenize(&tip.rule))
                    })
                    .collect();
                connected_clusters(&ids, &token_sets, threshold)
            };
            found.retain(|cluster| cluster.len() >= min_size);
            clusters.extend(found);
        }
        clusters.sort_by(|left, right| {
            right
                .len()
                .cmp(&left.len())
                .then_with(|| left.first().cmp(&right.first()))
        });
        clusters
    }

    /// Runs the purge pass and returns how many tips it removed.
    async fn purge(&mut self, bank: &mut TipBank) -> Result<usize> {
        let Some(model) = self.model.clone() else {
            return Ok(0);
        };
        let candidates = purge_candidates(bank, &self.form_checked);
        if candidates.is_empty() {
            return Ok(0);
        }
        let parser = TipParser::from_config(self.config());
        let defects: Vec<Option<&'static str>> = candidates
            .iter()
            .map(|(_, rule)| form_defect(rule, &parser))
            .collect();
        let mut verdicts: BTreeMap<usize, PurgeVerdict> = BTreeMap::new();
        let prompt = purge_prompt(&candidates);
        if let Some(answer) = invoke(&model, &self.request, DREAM_PURGE_INSTRUCTION, prompt).await {
            for verdict in parse_purge_verdicts(&answer, candidates.len()) {
                verdicts.insert(verdict.index, verdict);
            }
        }
        let mut purged = 0;
        for (index, (tip_id, _rule)) in candidates.iter().enumerate() {
            let purge = defects[index].is_some()
                || verdicts
                    .get(&index)
                    .is_some_and(|verdict| verdict.verdict == PurgeDecision::Purge);
            if purge {
                if bank.remove(tip_id).is_some() {
                    purged += 1;
                }
            } else if verdicts.contains_key(&index) {
                self.form_checked.insert(tip_id.clone());
            }
        }
        Ok(purged)
    }
}

/// The unchecked tips one purge prompt carries, bounded by count and characters.
fn purge_candidates(bank: &TipBank, checked: &BTreeSet<String>) -> Vec<(String, String)> {
    let mut candidates: Vec<(String, String)> = Vec::new();
    let mut used = 0;
    for tip in bank.tips() {
        if checked.contains(&tip.tip_id) {
            continue;
        }
        if candidates.len() >= PURGE_BATCH_SIZE {
            break;
        }
        let length = tip.rule.chars().count();
        if !candidates.is_empty() && used + length > PURGE_MAX_CHARS {
            break;
        }
        used += length;
        candidates.push((tip.tip_id.clone(), tip.rule.clone()));
    }
    candidates
}

/// Clusters `ids` by cosine similarity, skipping tips without a vector.
fn embedding_clusters(index: &EmbeddingIndex, ids: &[String], threshold: f64) -> Vec<Vec<String>> {
    let vectors: Vec<Option<&[f32]>> = ids.iter().map(|tip_id| index.vector(tip_id)).collect();
    let mut parents: Vec<usize> = (0..ids.len()).collect();
    for (left, left_vector) in vectors.iter().enumerate() {
        let Some(left_vector) = *left_vector else {
            continue;
        };
        for (right, right_vector) in vectors.iter().enumerate().skip(left + 1) {
            let Some(right_vector) = *right_vector else {
                continue;
            };
            if cosine(left_vector, right_vector) < threshold {
                continue;
            }
            let left_root = find(&mut parents, left);
            let right_root = find(&mut parents, right);
            if left_root == right_root {
                continue;
            }
            let (root, child) = if left_root < right_root {
                (left_root, right_root)
            } else {
                (right_root, left_root)
            };
            parents[child] = root;
        }
    }
    let mut clusters: Vec<Vec<String>> = Vec::new();
    let mut cluster_of_root: Vec<Option<usize>> = vec![None; ids.len()];
    for (position, tip_id) in ids.iter().enumerate() {
        if vectors[position].is_none() {
            continue;
        }
        let root = find(&mut parents, position);
        if let Some(cluster) = cluster_of_root[root] {
            clusters[cluster].push(tip_id.clone());
        } else {
            cluster_of_root[root] = Some(clusters.len());
            clusters.push(vec![tip_id.clone()]);
        }
    }
    clusters
}

/// Caps a cluster at `max` rules, keeping the highest-scoring ones.
fn capped_cluster(bank: &TipBank, cluster: &[String], max: usize) -> Vec<String> {
    if cluster.len() <= max {
        return cluster.to_vec();
    }
    let mut ordered: Vec<(String, f64)> = cluster
        .iter()
        .map(|tip_id| {
            let score = bank.get(tip_id).map_or(0.0, TipRecord::score);
            (tip_id.clone(), score)
        })
        .collect();
    ordered.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ordered.truncate(max);
    ordered.into_iter().map(|(tip_id, _)| tip_id).collect()
}

/// Pairwise similarities of `ids`, cosine when both vectors exist.
fn pairwise_similarities(
    bank: &TipBank,
    index: Option<&EmbeddingIndex>,
    ids: &[String],
) -> Vec<(usize, usize, f64)> {
    let mut similarities: Vec<(usize, usize, f64)> = Vec::new();
    for left in 0..ids.len() {
        for right in (left + 1)..ids.len() {
            let similarity = match index {
                Some(index) => match (index.vector(&ids[left]), index.vector(&ids[right])) {
                    (Some(left_vector), Some(right_vector)) => cosine(left_vector, right_vector),
                    _ => token_similarity(bank, &ids[left], &ids[right]),
                },
                None => token_similarity(bank, &ids[left], &ids[right]),
            };
            similarities.push((left, right, similarity));
        }
    }
    similarities
}

/// Token-set Jaccard similarity of two stored rules.
fn token_similarity(bank: &TipBank, left: &str, right: &str) -> f64 {
    let left_tokens = bank
        .get(left)
        .map_or_else(Vec::new, |tip| tokenize(&tip.rule));
    let right_tokens = bank
        .get(right)
        .map_or_else(Vec::new, |tip| tokenize(&tip.rule));
    jaccard(&left_tokens, &right_tokens)
}

/// Applies a parsed verdict, returning how many tips the merge absorbed.
fn apply_merge(bank: &mut TipBank, cluster: &[String], verdict: &MergeVerdict) -> Option<usize> {
    if verdict.decision == MergeDecision::KeepDistinct {
        return None;
    }
    let (merge, _keep) = normalize_merge_subset(
        cluster.len(),
        verdict.decision,
        &verdict.merge_indices,
        &verdict.keep_indices,
    );
    if merge.len() < 2 {
        return None;
    }
    let ids: Vec<String> = merge
        .iter()
        .filter_map(|index| cluster.get(*index).cloned())
        .collect();
    if ids.len() < 2 {
        return None;
    }
    let category: TipCategory = ids
        .iter()
        .find_map(|tip_id| bank.get(tip_id).map(|tip| tip.category))?;
    let canonical = verdict.canonical.trim();
    let canonical = if canonical.is_empty() {
        best_rule(bank, &ids)?
    } else {
        canonical.to_owned()
    };
    // An unusable canonical leaves the cluster distinct instead of deleting rules.
    TipRecord::new(canonical.clone(), category).ok()?;

    let representative = bank.merge_cluster(&ids)?;
    let tick = bank.tick();
    let current = bank.get(&representative)?.rule.clone();
    if current != canonical {
        let new_id = tip_id_for(category, &canonical);
        let collides = new_id != representative && bank.get(&new_id).is_some();
        if !collides {
            if let Some(record) = bank.get_mut(&representative) {
                record.rule = canonical;
                record.tip_id = new_id;
                "dream".clone_into(&mut record.source);
                record.updated_tick = tick;
            }
        }
    }
    Some(ids.len() - 1)
}

/// The rule of the highest-scoring member, ties by ascending `tip_id`.
fn best_rule(bank: &TipBank, ids: &[String]) -> Option<String> {
    ids.iter()
        .filter_map(|tip_id| bank.get(tip_id))
        .max_by(|left, right| {
            left.score()
                .total_cmp(&right.score())
                .then_with(|| right.tip_id.cmp(&left.tip_id))
        })
        .map(|tip| tip.rule.clone())
}

/// Sends one prompt and returns the answer text, logging a failed call.
async fn invoke(
    model: &Arc<dyn Model>,
    request: &ModelRequestConfig,
    instruction: &str,
    prompt: String,
) -> Option<String> {
    let messages = [BaseMessage::system(instruction), BaseMessage::user(prompt)];
    match model.invoke(&messages, request).await {
        Ok(response) => Some(
            response
                .message()
                .map_or_else(String::new, |message| message.text().into_owned()),
        ),
        Err(error) => {
            tracing::debug!(
                model = model.model_name(),
                error = %error,
                "ttse dream model call failed; keeping the current tips"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttse::dream_prompt::merge_answer;

    #[test]
    fn capped_clusters_keep_the_best_scoring_rules() {
        let mut bank = TipBank::with_capacity(8);
        for (rule, hits, successes, failures) in [
            ("rule one", 4, 0, 4),
            ("rule two", 4, 4, 0),
            ("rule three", 2, 1, 1),
        ] {
            let tip = TipRecord::new(rule, TipCategory::Tool)
                .expect("valid rule")
                .with_observation(hits, successes, failures);
            bank.add(tip);
        }
        let ids: Vec<String> = bank.tips().iter().map(|tip| tip.tip_id.clone()).collect();
        let capped = capped_cluster(&bank, &ids, 2);
        assert_eq!(capped.len(), 2);
        assert_eq!(capped[0], ids[1], "the highest success rate wins");
        assert_eq!(capped[1], ids[2]);
    }

    // ---- model-backed passes (offline `ScriptedModel`) ---------------------

    use std::sync::Arc;

    use async_trait::async_trait;
    use openjiuwen_common::error::Result as CrateResult;
    use openjiuwen_core::foundation::llm::{Model, ModelRequestConfig};
    use openjiuwen_core::retrieval::embedding::Embedding;
    use openjiuwen_test_support::ScriptedModel;

    use crate::ttse::config::TtseConfig;
    use crate::ttse::index::EmbeddingIndex;
    use crate::ttse::models::{TipCategory, TipRecord};

    /// Two rules whose token sets overlap well above the 0.72 dream threshold.
    const D1: &str = "always check the workspace path before running tests";
    const D2: &str = "always check the workspace path before running the tests";
    const D3: &str = "always check the workspace path before running all tests";
    /// A rule sharing no token with the duplicates.
    const UNRELATED: &str = "prefer rg over grep for speed";
    /// The canonical rule the scripted merge answer folds the duplicates into.
    const CANONICAL: &str = "When running tests: use cargo to run them offline";

    fn dream_config() -> TtseConfig {
        TtseConfig {
            dream_prune_enabled: false,
            dream_merge_threshold: 0.72,
            dream_cluster_min_size: 2,
            dream_ttl_ticks: 90,
            ..TtseConfig::default()
        }
    }

    fn bank_with(rules: &[(&str, TipCategory)]) -> TipBank {
        let mut bank = TipBank::with_capacity(16);
        bank.tick = 20;
        for (rule, category) in rules {
            bank.tips
                .push(TipRecord::new(*rule, *category).expect("valid rule"));
        }
        bank
    }

    fn scripted(answer: &str) -> Arc<dyn Model> {
        Arc::new(ScriptedModel::text(answer))
    }

    /// An embedder mapping every text to the same unit vector.
    struct ConstantEmbedding;

    #[async_trait]
    impl Embedding for ConstantEmbedding {
        fn model_name(&self) -> &'static str {
            "constant"
        }

        fn dimension_hint(&self) -> Option<usize> {
            Some(2)
        }

        async fn embed_query(&self, _text: &str) -> CrateResult<Vec<f32>> {
            Ok(vec![1.0, 0.0])
        }

        async fn embed_documents(
            &self,
            texts: &[String],
            _batch_size: Option<usize>,
        ) -> CrateResult<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }
    }

    #[tokio::test]
    async fn a_merge_verdict_collapses_a_cluster_into_the_canonical_text() {
        let mut dreamer = Dreamer::new(dream_config())
            .with_model(scripted(&merge_answer()), ModelRequestConfig::default());
        assert!(dreamer.has_model());
        let mut bank = bank_with(&[
            (D1, TipCategory::Tool),
            (D2, TipCategory::Tool),
            (D3, TipCategory::Tool),
            (UNRELATED, TipCategory::Tool),
        ]);

        let report = dreamer
            .dream_with_model(&mut bank, None)
            .await
            .expect("dream");

        assert_eq!(report.examined, 4);
        assert_eq!(report.merged, 1, "MERGE_INDICES folded two of the three");
        assert_eq!(report.promoted, 1);
        assert_eq!(bank.len(), 3);
        let canonical = bank
            .tips()
            .iter()
            .find(|tip| tip.rule == CANONICAL)
            .expect("the canonical rule replaced the merged members");
        assert_eq!(canonical.source, "dream");
        assert!(bank.tips().iter().any(|tip| tip.rule == UNRELATED));
    }

    #[tokio::test]
    async fn a_full_cluster_merge_absorbs_every_member() {
        let answer = [
            "THINKING:",
            "all three say the same thing",
            "REASON: paraphrases",
            "VERDICT: MERGE",
            &format!("CANONICAL: {CANONICAL}"),
            "MERGE_INDICES:",
            "KEEP_INDICES:",
        ]
        .join("\n");
        let mut dreamer = Dreamer::new(dream_config())
            .with_model(scripted(&answer), ModelRequestConfig::default());
        let mut bank = bank_with(&[
            (D1, TipCategory::Tool),
            (D2, TipCategory::Tool),
            (D3, TipCategory::Tool),
        ]);

        let report = dreamer
            .dream_with_model(&mut bank, None)
            .await
            .expect("dream");

        assert_eq!((report.merged, report.promoted), (2, 1));
        assert_eq!(bank.len(), 1);
        assert_eq!(bank.tips()[0].rule, CANONICAL);
    }

    #[tokio::test]
    async fn keep_distinct_leaves_the_bank_untouched() {
        let answer = [
            "THINKING:",
            "the conditions differ",
            "REASON: different cases",
            "VERDICT: KEEP_DISTINCT",
            "CANONICAL:",
        ]
        .join("\n");
        let mut dreamer = Dreamer::new(dream_config())
            .with_model(scripted(&answer), ModelRequestConfig::default());
        let mut bank = bank_with(&[(D1, TipCategory::Tool), (D2, TipCategory::Tool)]);
        let before = bank.clone();

        let report = dreamer
            .dream_with_model(&mut bank, None)
            .await
            .expect("dream");

        assert_eq!((report.merged, report.promoted, report.pruned), (0, 0, 0));
        assert_eq!(bank, before);
    }

    #[tokio::test]
    async fn malformed_model_output_keeps_the_tips() {
        let mut dreamer = Dreamer::new(dream_config()).with_model(
            scripted("I would merge those two"),
            ModelRequestConfig::default(),
        );
        let mut bank = bank_with(&[(D1, TipCategory::Tool), (D2, TipCategory::Tool)]);
        let before = bank.clone();

        let report = dreamer
            .dream_with_model(&mut bank, None)
            .await
            .expect("dream never errors on a bad answer");

        assert_eq!(report.merged, 0);
        assert_eq!(bank, before);
    }

    async fn index_of(bank: &TipBank) -> EmbeddingIndex {
        let mut index = EmbeddingIndex::new(Arc::new(ConstantEmbedding));
        for tip in bank.tips() {
            index.upsert(&tip.tip_id, &tip.rule).await.expect("upsert");
        }
        index
    }
    #[tokio::test]
    async fn cross_category_clusters_never_merge() {
        let mut dreamer = Dreamer::new(dream_config())
            .with_model(scripted(&merge_answer()), ModelRequestConfig::default());
        let mut bank = bank_with(&[(D1, TipCategory::Tool), (D2, TipCategory::Pitfall)]);
        let index = index_of(&bank).await;
        let before = bank.clone();

        let report = dreamer
            .dream_with_model(&mut bank, Some(&index))
            .await
            .expect("dream");

        assert_eq!(report.merged, 0);
        assert_eq!(bank, before, "identical vectors never cross a category");
    }

    #[tokio::test]
    async fn cosine_clustering_merges_rules_token_overlap_misses() {
        let first = "use rg instead of grep";
        let second = "prefer ripgrep over grep";
        let mut dreamer = Dreamer::new(dream_config())
            .with_model(scripted(&merge_answer()), ModelRequestConfig::default());
        let mut bank = bank_with(&[(first, TipCategory::Tool), (second, TipCategory::Tool)]);
        let index = index_of(&bank).await;

        let report = dreamer
            .dream_with_model(&mut bank, Some(&index))
            .await
            .expect("dream");

        assert_eq!((report.merged, report.promoted), (1, 1));
        assert_eq!(bank.len(), 1);
        assert_eq!(bank.tips()[0].rule, CANONICAL);
    }

    #[tokio::test]
    async fn purge_verdicts_remove_the_flagged_tip() {
        let answer = [
            "INDEX: 0 | VERDICT: PURGE | REASON: tip_malformed",
            "INDEX: 1 | VERDICT: KEEP | REASON: ok",
            "INDEX: 2 | VERDICT: KEEP | REASON: ok",
        ]
        .join("\n");
        let mut dreamer = Dreamer::new(dream_config())
            .with_model(scripted(&answer), ModelRequestConfig::default());
        let first = "use rg instead of grep";
        let mut bank = bank_with(&[
            (first, TipCategory::Tool),
            ("run the tests offline", TipCategory::Workflow),
            ("quote every path", TipCategory::Pitfall),
        ]);

        let report = dreamer
            .dream_with_model(&mut bank, None)
            .await
            .expect("dream");

        assert_eq!(report.pruned, 1);
        assert_eq!(bank.len(), 2);
        assert!(bank.tips().iter().all(|tip| tip.rule != first));
    }

    #[tokio::test]
    async fn the_deterministic_fallback_purges_a_generic_rule() {
        let answer = [
            "INDEX: 0 | VERDICT: KEEP | REASON: ok",
            "INDEX: 1 | VERDICT: KEEP | REASON: ok",
        ]
        .join("\n");
        let mut dreamer = Dreamer::new(dream_config())
            .with_model(scripted(&answer), ModelRequestConfig::default());
        let mut bank = bank_with(&[
            ("check", TipCategory::Tool),
            ("use rg instead of grep", TipCategory::Tool),
        ]);

        let report = dreamer
            .dream_with_model(&mut bank, None)
            .await
            .expect("dream");

        assert_eq!(report.pruned, 1, "the one-word rule is not actionable");
        assert_eq!(bank.len(), 1);
        assert_eq!(bank.tips()[0].rule, "use rg instead of grep");
    }

    #[tokio::test]
    async fn the_no_model_path_equals_the_deterministic_dream() {
        let mut plain = Dreamer::new(dream_config());
        let mut model_free = Dreamer::new(dream_config());
        let rules = [
            (D1, TipCategory::Tool),
            (D2, TipCategory::Tool),
            (D3, TipCategory::Tool),
            (UNRELATED, TipCategory::Tool),
        ];
        let mut left = bank_with(&rules);
        let mut right = bank_with(&rules);

        let expected = plain.dream(&mut left);
        let actual = model_free
            .dream_with_model(&mut right, None)
            .await
            .expect("dream");

        assert_eq!(actual, expected);
        assert_eq!(right, left);
    }

    #[tokio::test]
    async fn the_merge_budget_caps_the_number_of_prompts() {
        let model = ScriptedModel::text(merge_answer());
        let mut dreamer = Dreamer::new(dream_config()).with_max_llm_merges(0);
        dreamer = dreamer.with_model(
            Arc::new(model.clone()) as Arc<dyn Model>,
            ModelRequestConfig::default(),
        );
        assert_eq!(dreamer.max_llm_merges(), 0);
        let mut bank = bank_with(&[(D1, TipCategory::Tool), (D2, TipCategory::Tool)]);

        let report = dreamer
            .dream_with_model(&mut bank, None)
            .await
            .expect("dream");

        assert_eq!(report.merged, 0);
        assert_eq!(bank.len(), 2);
    }
}
