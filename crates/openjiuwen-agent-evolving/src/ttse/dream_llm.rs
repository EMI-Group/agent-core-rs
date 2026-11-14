//! Model-backed dreaming: LLM cluster merges and the tip purge pass.
//!
//! Ports the LLM half of Python `openjiuwen.agent_evolving.ttse.dream` and its
//! prompts (`ttse/prompts.py`): clusters of near-duplicate tips are consolidated
//! by a model that answers `THINKING / REASON / VERDICT / CANONICAL /
//! MERGE_INDICES / KEEP_INDICES`, and a second prompt quality-checks tip form,
//! answering one `INDEX: i | VERDICT: KEEP|PURGE | REASON: …` line per tip.
//!
//! Everything the model says is advisory: an unparseable answer leaves the
//! cluster distinct, a blank or unusable `CANONICAL` is rejected, and clusters
//! are always built inside one [`TipCategory`], so a merge can never fold rules
//! from different categories together. The deterministic
//! [`Dreamer::dream`](crate::ttse::Dreamer::dream) pass remains the fallback and
//! the no-model path of [`Dreamer::dream_with_model`].

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_core::foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_core::foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};

use crate::ttse::bank::{jaccard, tokenize, TipBank};
use crate::ttse::dream::{connected_clusters, find, merge_threshold, Dreamer, MIN_CLUSTER_SIZE};
use crate::ttse::index::{cosine, EmbeddingIndex};
use crate::ttse::models::{tip_id_for, DreamReport, TipCategory, TipRecord};
use crate::ttse::parse::TipParser;

/// How many merge prompts one model-backed pass sends by default (Python's
/// `dream_max_llm_merges`).
pub const DEFAULT_MAX_LLM_MERGES: usize = 15;
/// How many rules one merge prompt carries; the highest-scoring ones win.
const MAX_PROMPT_RULES: usize = 12;
/// How many tips one purge prompt carries.
const PURGE_BATCH_SIZE: usize = 24;
/// How many rule characters one purge prompt carries.
const PURGE_MAX_CHARS: usize = 4_000;
/// A single-word rule shorter than this is treated as too generic to act on.
const MIN_RULE_CHARS: usize = 8;

/// The merge system message, ported from `prompts.py::DREAM_MERGE_SYSTEM`.
pub const DREAM_MERGE_INSTRUCTION: &str = "You consolidate near-duplicate rules in a single track of an experience bank. Reduce redundancy without dropping mutually exclusive conditions. Always write a multi-line THINKING chain of thought first, then REASON, then the verdict. Output the structured verdict format exactly.";

/// The purge system message, ported from `prompts.py::DREAM_PURGE_SYSTEM`.
pub const DREAM_PURGE_INSTRUCTION: &str = "You quality-check TIP rules in an agent experience bank. Judge format validity and over-genericity only. Do not require capabilities to appear on any whitelist. Output one structured line per tip index exactly.";

/// The verdict one merge prompt returned.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeDecision {
    /// Fold the members into one canonical rule.
    Merge,
    /// Replace the members with a rewritten canonical rule.
    Rewrite,
    /// Leave every member untouched; also the safe fallback.
    #[default]
    KeepDistinct,
}

impl MergeDecision {
    /// The stable `snake_case` label used by JSON and diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Merge => "merge",
            Self::Rewrite => "rewrite",
            Self::KeepDistinct => "keep_distinct",
        }
    }
}

/// A parsed merge reply.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MergeVerdict {
    /// What the model decided.
    pub decision: MergeDecision,
    /// The retained or rewritten rule; empty means "use the best member".
    pub canonical: String,
    /// The zero-based members folded into the canonical; empty means all.
    pub merge_indices: Vec<usize>,
    /// The zero-based members left untouched; audit-only for `KEEP_DISTINCT`.
    pub keep_indices: Vec<usize>,
    /// The one-sentence decision summary.
    pub reason: String,
    /// The chain of thought the model wrote first.
    pub thinking: String,
}

/// What the purge pass decided for one tip.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PurgeDecision {
    /// The rule is usable.
    #[default]
    Keep,
    /// The rule is malformed or too generic and should be dropped.
    Purge,
}

/// A parsed purge line.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PurgeVerdict {
    /// The zero-based index of the tip in the prompt.
    pub index: usize,
    /// What the model decided.
    pub verdict: PurgeDecision,
    /// One of the documented reason labels, or `ok`.
    pub reason: String,
}

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
            let mut found = match index {
                Some(index) => embedding_clusters(index, &ids, threshold),
                None => {
                    let token_sets: Vec<Vec<String>> = ids
                        .iter()
                        .map(|tip_id| {
                            bank.get(tip_id)
                                .map_or_else(Vec::new, |tip| tokenize(&tip.rule))
                        })
                        .collect();
                    connected_clusters(&ids, &token_sets, threshold)
                }
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

/// Parses a `THINKING / REASON / VERDICT / CANONICAL / *_INDICES` reply.
///
/// `THINKING`, `REASON` and a recognised `VERDICT` are mandatory; anything else
/// returns `None`, which leaves the cluster distinct. The parse is tolerant of
/// spacing, inline `THINKING: …` bodies and reordered trailing fields.
#[must_use]
pub fn parse_merge_verdict(text: &str, cluster_size: usize) -> Option<MergeVerdict> {
    if text.trim().is_empty() {
        return None;
    }
    let mut decision: Option<MergeDecision> = None;
    let mut canonical = String::new();
    let mut merge_indices: Vec<usize> = Vec::new();
    let mut keep_indices: Vec<usize> = Vec::new();
    let mut reason = String::new();
    let mut thinking: Vec<String> = Vec::new();
    let mut in_thinking = false;
    for line in text.lines() {
        let trimmed = line.trim();
        let upper = trimmed.to_uppercase();
        if upper.starts_with("THINKING") {
            in_thinking = true;
            if let Some(body) = payload(trimmed) {
                if !body.is_empty() {
                    thinking.push(body.to_owned());
                }
            }
            continue;
        }
        if in_thinking {
            if !is_merge_header(trimmed) {
                thinking.push(line.trim_end().to_owned());
                continue;
            }
            in_thinking = false;
        }
        if upper.starts_with("VERDICT") {
            decision = payload(trimmed).and_then(|body| {
                match body
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .to_uppercase()
                    .as_str()
                {
                    "MERGE" => Some(MergeDecision::Merge),
                    "REWRITE" => Some(MergeDecision::Rewrite),
                    "KEEP_DISTINCT" => Some(MergeDecision::KeepDistinct),
                    _ => None,
                }
            });
        } else if upper.starts_with("CANONICAL") {
            canonical = payload(trimmed).unwrap_or("").to_owned();
        } else if upper.starts_with("MERGE_INDICES") {
            merge_indices = index_list(payload(trimmed).unwrap_or(""), cluster_size);
        } else if upper.starts_with("KEEP_INDICES") {
            keep_indices = index_list(payload(trimmed).unwrap_or(""), cluster_size);
        } else if upper.starts_with("REASON") {
            reason = payload(trimmed).unwrap_or(trimmed).to_owned();
        }
    }
    let thinking = thinking.join("\n").trim().to_owned();
    let decision = decision?;
    if thinking.is_empty() || reason.is_empty() {
        return None;
    }
    Some(MergeVerdict {
        decision,
        canonical,
        merge_indices,
        keep_indices,
        reason,
        thinking,
    })
}

/// Resolves a verdict's subset into `(merge, keep)` member indices.
///
/// `KEEP_DISTINCT` merges nothing and keeps every member. For `MERGE`/`REWRITE`
/// an empty `merge_indices` means the whole cluster; unlisted members are kept,
/// and a subset smaller than two degrades to keeping everything.
#[must_use]
pub fn normalize_merge_subset(
    cluster_size: usize,
    decision: MergeDecision,
    merge_indices: &[usize],
    keep_indices: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let all: Vec<usize> = (0..cluster_size).collect();
    if decision == MergeDecision::KeepDistinct {
        return (Vec::new(), all);
    }
    let merge = dedupe_indices(
        merge_indices
            .iter()
            .copied()
            .filter(|index| *index < cluster_size)
            .collect(),
    );
    if merge.is_empty() {
        return (all, Vec::new());
    }
    let mut keep = dedupe_indices(
        keep_indices
            .iter()
            .copied()
            .filter(|index| *index < cluster_size)
            .collect(),
    );
    keep.retain(|index| !merge.contains(index));
    if keep_indices.is_empty() {
        keep = all
            .iter()
            .copied()
            .filter(|index| !merge.contains(index))
            .collect();
    } else {
        for index in &all {
            if !merge.contains(index) && !keep.contains(index) {
                keep.push(*index);
            }
        }
        keep = dedupe_indices(keep);
    }
    if merge.len() < 2 {
        return (Vec::new(), all);
    }
    (merge, keep)
}

/// Parses `INDEX: i | VERDICT: KEEP|PURGE | REASON: …` lines.
///
/// Lines without an in-range index or a recognised verdict are ignored, so a
/// partial reply still applies where it is unambiguous.
#[must_use]
pub fn parse_purge_verdicts(text: &str, batch_size: usize) -> Vec<PurgeVerdict> {
    let mut verdicts: Vec<PurgeVerdict> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let upper = trimmed.to_uppercase();
        if !upper.contains("INDEX") || !upper.contains("VERDICT") {
            continue;
        }
        let mut index: Option<usize> = None;
        let mut decision: Option<PurgeDecision> = None;
        let mut reason = String::new();
        for part in trimmed.replace('|', " | ").split('|') {
            let part = part.trim();
            let upper = part.to_uppercase();
            if upper.starts_with("INDEX") {
                index = payload(part).and_then(|body| {
                    body.split_whitespace()
                        .next()
                        .and_then(|token| token.parse().ok())
                });
            } else if upper.starts_with("VERDICT") {
                decision = payload(part).and_then(|body| {
                    match body
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .to_uppercase()
                        .as_str()
                    {
                        "KEEP" => Some(PurgeDecision::Keep),
                        "PURGE" => Some(PurgeDecision::Purge),
                        _ => None,
                    }
                });
            } else if upper.starts_with("REASON") {
                reason = payload(part).unwrap_or("").to_owned();
            }
        }
        let (Some(index), Some(verdict)) = (index, decision) else {
            continue;
        };
        if index >= batch_size {
            continue;
        }
        if reason.is_empty() {
            reason = "ok".to_owned();
        }
        verdicts.push(PurgeVerdict {
            index,
            verdict,
            reason,
        });
    }
    verdicts
}

/// The deterministic form defect of `rule`, when there is one.
///
/// This is the fallback the purge pass applies even when the model keeps a rule:
/// blank rules, text the [`TipParser`] rejects, near-empty tokens and single-word
/// rules shorter than [`MIN_RULE_CHARS`] are not actionable advice. Words are
/// whitespace-delimited, so a short unsplit CJK rule is judged by character
/// count only.
fn form_defect(rule: &str, parser: &TipParser) -> Option<&'static str> {
    let trimmed = rule.trim();
    if trimmed.is_empty() || parser.parse(trimmed).is_empty() {
        return Some("tip_malformed");
    }
    if trimmed
        .chars()
        .filter(|character| character.is_alphanumeric())
        .count()
        < 3
    {
        return Some("tip_malformed");
    }
    if trimmed.split_whitespace().count() < 2 && trimmed.chars().count() < MIN_RULE_CHARS {
        return Some("tip_too_generic_action");
    }
    None
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
    for left in 0..ids.len() {
        let Some(left_vector) = vectors[left] else {
            continue;
        };
        for right in (left + 1)..ids.len() {
            let Some(right_vector) = vectors[right] else {
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

/// Builds the per-cluster merge prompt around the numbered rules and similarity
/// table (see the [module docs](self) for the expected answer).
fn merge_prompt(bank: &TipBank, ids: &[String], similarities: &[(usize, usize, f64)]) -> String {
    let rules: Vec<String> = ids
        .iter()
        .enumerate()
        .map(|(index, tip_id)| {
            let tip = bank.get(tip_id);
            let rule = tip.map_or("", |tip| tip.rule.as_str());
            let count = tip.map_or(0, |tip| tip.hits);
            format!("{index}. {rule} (count={count})")
        })
        .collect();
    let sim_table = if similarities.is_empty() {
        "(none)".to_owned()
    } else {
        similarities
            .iter()
            .map(|(left, right, similarity)| format!("{left},{right},{similarity:.3}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "You are consolidating a cluster of near-duplicate TIP rules from an agent experience bank.\n\n\
         These rules already share the same business-scenario category. Goal: reduce redundancy while\n\
         preserving distinct conditions. Prefer MERGE or REWRITE when rules are paraphrases of the\n\
         same idea; KEEP_DISTINCT when conditions conflict or cover different cases. The count field\n\
         is only an importance hint — never override \"different conditions\" just because one count\n\
         is higher.\n\n\
         At most ONE merge per reply: fold one subset into a single CANONICAL; leave the rest\n\
         unchanged. Do not emit multiple canonicals.\n\n\
         TIP constraints:\n\
         - CANONICAL (for MERGE/REWRITE) MUST be exactly: When <condition>: use <capability> to <action>\n\
         - If conditions are mutually exclusive or meaningfully different, choose KEEP_DISTINCT.\n\
         - Do NOT invent capabilities.\n\n\
         Cluster rules (0-based index, text, count):\n\
         {}\n\n\
         Pairwise cosine similarities (i, j, sim):\n\
         {sim_table}\n\n\
         Reply in EXACTLY this format (field order mandatory):\n\
         THINKING:\n\
         <multi-line chain of thought: compare members, similarities, and conditions>\n\
         REASON: <one-sentence decision summary>\n\
         VERDICT: MERGE | KEEP_DISTINCT | REWRITE\n\
         CANONICAL: <single retained or rewritten text; empty allowed for KEEP_DISTINCT>\n\
         MERGE_INDICES: <comma-separated 0-based indices folded into CANONICAL; empty = all>\n\
         KEEP_INDICES: <comma-separated 0-based indices left unchanged; empty = complement>\n",
        rules.join("\n")
    )
}

/// Builds the purge prompt around the numbered candidate rules.
fn purge_prompt(candidates: &[(String, String)]) -> String {
    let tips: Vec<String> = candidates
        .iter()
        .enumerate()
        .map(|(index, (_, rule))| format!("{index}. {rule}"))
        .collect();
    format!(
        "You are reviewing TIP rules from an agent experience bank.\n\n\
         For each tip, decide KEEP or PURGE using ONLY these criteria:\n\
         1. Format: a valid TIP is shaped like When <condition>: use <capability> to <action>.\n\
         2. Over-generic: purge when the condition is empty/vacuous or the action is vague.\n\
         3. Do NOT purge merely because a capability is unfamiliar or not on a list.\n\n\
         Tips (0-based index):\n\
         {}\n\n\
         Reply with EXACTLY one line per tip, in index order, no other text:\n\
         INDEX: <i> | VERDICT: KEEP|PURGE | REASON: tip_malformed|tip_fact_shaped|tip_too_generic_condition|tip_too_generic_action|ok\n",
        tips.join("\n")
    )
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
                record.source = "dream".to_owned();
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

/// The payload of a `HEADER: body` line, trimmed; `None` without a colon.
fn payload(line: &str) -> Option<&str> {
    line.split_once(':').map(|(_, body)| body.trim())
}

/// Whether a line starts one of the structured merge fields.
fn is_merge_header(line: &str) -> bool {
    let upper = line.to_uppercase();
    [
        "THINKING",
        "REASON",
        "VERDICT",
        "CANONICAL",
        "MERGE_INDICES",
        "KEEP_INDICES",
    ]
    .iter()
    .any(|header| upper.starts_with(header))
}

/// The deduped, in-range indices a `MERGE_INDICES`/`KEEP_INDICES` payload lists.
fn index_list(payload: &str, cluster_size: usize) -> Vec<usize> {
    dedupe_indices(
        payload
            .replace(',', " ")
            .split_whitespace()
            .filter_map(|token| token.parse::<usize>().ok())
            .filter(|index| *index < cluster_size)
            .collect(),
    )
}

/// Removes duplicates while preserving first-seen order.
fn dedupe_indices(indices: Vec<usize>) -> Vec<usize> {
    let mut seen: BTreeMap<usize, ()> = BTreeMap::new();
    let mut deduped: Vec<usize> = Vec::new();
    for index in indices {
        if seen.insert(index, ()).is_none() {
            deduped.push(index);
        }
    }
    deduped
}

#[cfg(test)]
mod tests {
    use super::*;

    fn merge_answer() -> String {
        [
            "THINKING:",
            "the first two rules say the same thing",
            "REASON: they are paraphrases",
            "VERDICT: MERGE",
            "CANONICAL: When running tests: use cargo to run them offline",
            "MERGE_INDICES: 0,1",
            "KEEP_INDICES: 2",
        ]
        .join("\n")
    }

    #[test]
    fn merge_verdict_parses_every_field() {
        let verdict = parse_merge_verdict(&merge_answer(), 3).expect("parsed");
        assert_eq!(verdict.decision, MergeDecision::Merge);
        assert_eq!(verdict.merge_indices, vec![0, 1]);
        assert_eq!(verdict.keep_indices, vec![2]);
        assert_eq!(verdict.reason, "they are paraphrases");
        assert!(verdict.thinking.contains("paraphrases"));
        assert!(verdict.canonical.starts_with("When running tests"));
    }

    #[test]
    fn malformed_merge_answers_are_rejected() {
        assert!(parse_merge_verdict("", 2).is_none());
        assert!(parse_merge_verdict("I merged them", 2).is_none());
        assert!(
            parse_merge_verdict("THINKING:\nreasoning\nVERDICT: MAYBE\nREASON: unsure", 2)
                .is_none()
        );
        assert!(
            parse_merge_verdict("THINKING:\nreasoning\nVERDICT: MERGE\nCANONICAL: x", 2).is_none(),
            "a missing REASON is malformed"
        );
        assert!(parse_merge_verdict("VERDICT: MERGE\nREASON: r", 2).is_none());
    }

    #[test]
    fn inline_thinking_and_ignored_indices_are_tolerated() {
        let answer = "THINKING: they overlap\nREASON: overlap\nVERDICT: KEEP_DISTINCT\nMERGE_INDICES: 5, 1, 1";
        let verdict = parse_merge_verdict(answer, 2).expect("parsed");
        assert_eq!(verdict.decision, MergeDecision::KeepDistinct);
        assert_eq!(verdict.merge_indices, vec![1]);
        assert_eq!(verdict.thinking, "they overlap");
    }

    #[test]
    fn subset_normalization_matches_python() {
        assert_eq!(
            normalize_merge_subset(3, MergeDecision::KeepDistinct, &[0], &[1]),
            (Vec::new(), vec![0, 1, 2])
        );
        assert_eq!(
            normalize_merge_subset(3, MergeDecision::Merge, &[], &[]),
            (vec![0, 1, 2], Vec::new())
        );
        assert_eq!(
            normalize_merge_subset(3, MergeDecision::Merge, &[0, 1], &[2]),
            (vec![0, 1], vec![2])
        );
        assert_eq!(
            normalize_merge_subset(3, MergeDecision::Rewrite, &[0, 1], &[]),
            (vec![0, 1], vec![2])
        );
        assert_eq!(
            normalize_merge_subset(3, MergeDecision::Merge, &[0], &[]),
            (Vec::new(), vec![0, 1, 2]),
            "a one-member merge keeps everything"
        );
        assert_eq!(
            normalize_merge_subset(3, MergeDecision::Merge, &[0, 1], &[3]),
            (vec![0, 1], vec![2]),
            "unlisted members are kept"
        );
    }

    #[test]
    fn purge_verdicts_parse_and_ignore_out_of_range_lines() {
        let answer = [
            "INDEX: 0 | VERDICT: KEEP | REASON: ok",
            "INDEX:1|VERDICT:PURGE|REASON:tip_malformed",
            "junk line",
            "INDEX: 9 | VERDICT: PURGE | REASON: tip_malformed",
            "INDEX: 2 | VERDICT: MAYBE | REASON: ok",
        ]
        .join("\n");
        let verdicts = parse_purge_verdicts(&answer, 3);
        assert_eq!(verdicts.len(), 2);
        assert_eq!(verdicts[0].index, 0);
        assert_eq!(verdicts[0].verdict, PurgeDecision::Keep);
        assert_eq!(verdicts[1].index, 1);
        assert_eq!(verdicts[1].verdict, PurgeDecision::Purge);
        assert_eq!(verdicts[1].reason, "tip_malformed");
        assert!(parse_purge_verdicts("", 3).is_empty());
    }

    #[test]
    fn form_defects_flag_malformed_and_generic_rules() {
        let parser = TipParser::default();
        assert_eq!(form_defect("   ", &parser), Some("tip_malformed"));
        assert_eq!(form_defect("!!", &parser), Some("tip_malformed"));
        assert_eq!(
            form_defect("check", &parser),
            Some("tip_too_generic_action")
        );
        assert_eq!(form_defect("use rg instead of grep", &parser), None);
        assert_eq!(form_defect("运行命令前检查工作区路径", &parser), None);
    }

    #[test]
    fn capped_clusters_keep_the_best_scoring_rules() {
        let mut bank = TipBank::with_capacity(8);
        for (rule, hits) in [("rule one", 0), ("rule two", 4), ("rule three", 2)] {
            let tip = TipRecord::new(rule, TipCategory::Tool)
                .expect("valid rule")
                .with_observation(hits, hits, 0);
            bank.add(tip);
        }
        let ids: Vec<String> = bank.tips().iter().map(|tip| tip.tip_id.clone()).collect();
        let capped = capped_cluster(&bank, &ids, 2);
        assert_eq!(capped.len(), 2);
        assert_eq!(capped[0], ids[1], "the best-observed rule wins");
        assert_eq!(capped[1], ids[2]);
    }
}
