//! The tip bank: dedupe, ranking, observation bookkeeping and pruning.
//!
//! Ports the deterministic part of `ttse/stores.py` and `ttse/index.py`: a
//! compact in-process record store keyed by [`TipRecord::tip_id`]. Python backs
//! the store with JSON on disk and an embedding index; here similarity is a
//! token-set Jaccard score, which is deterministic, offline and needs no model.
//!
//! Two helpers are shared with the induction, consultation and dream passes of
//! later waves: [`tokenize`] (ASCII runs plus one token per CJK character) and
//! [`jaccard`] (set similarity). Both are pure and stable, so bank contents do
//! not depend on hash iteration order.
//!
//! The bank never reads a clock: [`TipBank::increment_tick`] is the only source
//! of time, and callers drive it.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::ttse::config::TtseConfig;
use crate::ttse::models::{normalize_keywords, TipCategory, TipRecord};

/// Capacity applied by [`TipBank::default`].
const DEFAULT_CAPACITY: usize = 512;
/// Dedupe threshold applied by [`TipBank::default`].
const DEFAULT_DEDUP_THRESHOLD: f64 = 0.88;
/// Tips lighter than this are pruned by [`TipBank::prune`].
pub(crate) const MIN_WEIGHT: f64 = 0.05;

/// The result of storing one tip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AddOutcome {
    /// The tip was stored under the given id.
    Inserted(String),
    /// The tip was similar enough to an existing one to be merged into it.
    Merged {
        /// The id of the record that absorbed the incoming tip.
        into: String,
        /// The similarity that triggered the merge.
        similarity: f64,
    },
}

impl AddOutcome {
    /// The id of the record the bank holds after the operation.
    #[must_use]
    pub fn tip_id(&self) -> &str {
        match self {
            Self::Inserted(tip_id) | Self::Merged { into: tip_id, .. } => tip_id,
        }
    }
}

/// The tip bank: the store every TTSE pass reads and writes.
///
/// Unknown JSON fields are ignored and missing fields fall back to
/// [`TipBank::default`], so bank files are forward compatible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TipBank {
    /// Maximum number of tips kept; the lowest-scoring tips are evicted.
    pub capacity: usize,
    /// Jaccard similarity at or above which an incoming tip is merged.
    pub dedup_threshold: f64,
    /// The stored tips, in insertion order.
    pub tips: Vec<TipRecord>,
    /// The monotonic bank tick, advanced by [`TipBank::increment_tick`].
    pub tick: u64,
}

impl Default for TipBank {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_CAPACITY,
            dedup_threshold: DEFAULT_DEDUP_THRESHOLD,
            tips: Vec::new(),
            tick: 0,
        }
    }
}

impl TipBank {
    /// Builds a bank, normalizing the capacity to at least `1` and clamping the
    /// dedupe threshold to `0.0..=1.0`.
    ///
    /// A non-finite threshold falls back to the default instead of poisoning
    /// every later comparison.
    #[must_use]
    pub fn new(capacity: usize, dedup_threshold: f64) -> Self {
        let dedup_threshold = if dedup_threshold.is_finite() {
            dedup_threshold.clamp(0.0, 1.0)
        } else {
            DEFAULT_DEDUP_THRESHOLD
        };
        Self {
            capacity: capacity.max(1),
            dedup_threshold,
            tips: Vec::new(),
            tick: 0,
        }
    }

    /// Builds an empty bank with the default dedupe threshold.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(capacity, DEFAULT_DEDUP_THRESHOLD)
    }

    /// Stores a tip, merging it into the most similar existing record when the
    /// similarity reaches [`Self::dedup_threshold`].
    ///
    /// The scan is deterministic: the highest similarity wins and ties go to the
    /// lexicographically smallest existing `tip_id`. Inserting stamps the ticks
    /// and evicts the lowest `(score(), hits, tip_id)` records until the bank
    /// fits its capacity.
    pub fn add(&mut self, tip: TipRecord) -> AddOutcome {
        let tokens = tokenize(&tip.rule);
        let mut best: Option<(f64, String)> = None;
        for existing in &self.tips {
            let similarity = jaccard(&tokens, &tokenize(&existing.rule));
            if similarity < self.dedup_threshold {
                continue;
            }
            let better = match &best {
                None => true,
                Some((best_similarity, best_id)) => match similarity.total_cmp(best_similarity) {
                    Ordering::Greater => true,
                    Ordering::Equal => existing.tip_id < *best_id,
                    Ordering::Less => false,
                },
            };
            if better {
                best = Some((similarity, existing.tip_id.clone()));
            }
        }
        if let Some((similarity, into)) = best {
            if let Some(index) = self
                .tips
                .iter()
                .position(|existing| existing.tip_id == into)
            {
                merge_record(&mut self.tips[index], &tip, self.tick);
                self.enforce_capacity();
                return AddOutcome::Merged { into, similarity };
            }
        }
        let mut incoming = tip;
        if incoming.created_tick == 0 {
            incoming.created_tick = self.tick;
        }
        incoming.updated_tick = self.tick;
        let tip_id = incoming.tip_id.clone();
        self.tips.push(incoming);
        self.enforce_capacity();
        AddOutcome::Inserted(tip_id)
    }

    /// Advances the bank tick and returns the new value.
    pub fn increment_tick(&mut self) -> u64 {
        self.tick = self.tick.saturating_add(1);
        self.tick
    }

    /// The current bank tick.
    #[must_use]
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Records one observation for a tip; returns whether the tip exists.
    pub fn observe(&mut self, tip_id: &str, success: bool) -> bool {
        let tick = self.tick;
        let Some(tip) = self.tips.iter_mut().find(|tip| tip.tip_id == tip_id) else {
            return false;
        };
        tip.hits = tip.hits.saturating_add(1);
        if success {
            tip.successes = tip.successes.saturating_add(1);
        } else {
            tip.failures = tip.failures.saturating_add(1);
        }
        tip.updated_tick = tick;
        true
    }

    /// Looks up a tip by id.
    #[must_use]
    pub fn get(&self, tip_id: &str) -> Option<&TipRecord> {
        self.tips.iter().find(|tip| tip.tip_id == tip_id)
    }

    /// Looks up a tip by id, mutably.
    #[must_use]
    pub fn get_mut(&mut self, tip_id: &str) -> Option<&mut TipRecord> {
        self.tips.iter_mut().find(|tip| tip.tip_id == tip_id)
    }

    /// Removes a tip by id and returns it.
    pub fn remove(&mut self, tip_id: &str) -> Option<TipRecord> {
        let index = self.tips.iter().position(|tip| tip.tip_id == tip_id)?;
        Some(self.tips.remove(index))
    }

    /// The number of stored tips.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tips.len()
    }

    /// Whether the bank holds no tips.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tips.is_empty()
    }

    /// The stored tips, in insertion order.
    #[must_use]
    pub fn tips(&self) -> &[TipRecord] {
        &self.tips
    }

    /// The `k` best tips, cloned and ranked by descending effective score then
    /// ascending `tip_id`.
    ///
    /// With a query, the effective score is `score() * (1.0 + overlap)` where the
    /// overlap is the Jaccard similarity between the query tokens and the tokens
    /// of the rule plus keywords. Ranking is deterministic: no hash ordering, no
    /// tie broken by insertion time.
    #[must_use]
    pub fn top(&self, k: usize, query: Option<&str>) -> Vec<TipRecord> {
        let query_tokens = query.map(tokenize);
        let mut ranked: Vec<(TipRecord, f64)> = self
            .tips
            .iter()
            .map(|tip| {
                let effective = match &query_tokens {
                    Some(tokens) => {
                        let haystack = format!("{} {}", tip.rule, tip.keywords.join(" "));
                        tip.score() * (1.0 + jaccard(tokens, &tokenize(&haystack)))
                    }
                    None => tip.score(),
                };
                (tip.clone(), effective)
            })
            .collect();
        ranked.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.tip_id.cmp(&right.0.tip_id))
        });
        ranked.into_iter().take(k).map(|(tip, _)| tip).collect()
    }

    /// The tips of one category, in stored order.
    #[must_use]
    pub fn by_category(&self, category: TipCategory) -> Vec<&TipRecord> {
        self.tips
            .iter()
            .filter(|tip| tip.category == category)
            .collect()
    }

    /// Removes tips untouched for more than `dream_ttl_ticks` ticks, plus tips
    /// whose weight fell below the `MIN_WEIGHT` floor.
    ///
    /// With `dream_ttl_ticks == 0` only tips untouched since the current tick are
    /// prunable. Returns the number of removed tips.
    pub fn prune(&mut self, config: &TtseConfig) -> usize {
        let before = self.tips.len();
        let tick = self.tick;
        let ttl = config.dream_ttl_ticks;
        self.tips.retain(|tip| {
            let stale = tick.saturating_sub(tip.updated_tick) > ttl;
            !stale && tip.weight >= MIN_WEIGHT
        });
        before - self.tips.len()
    }

    /// Multiplies every weight by `factor`; an invalid factor is a no-op.
    ///
    /// Weights stay non-negative, so decay can never make a tip invalid.
    pub fn decay(&mut self, factor: f64) {
        if !factor.is_finite() || factor < 0.0 {
            return;
        }
        for tip in &mut self.tips {
            tip.weight = (tip.weight * factor).max(0.0);
        }
    }

    /// Aggregate counts over the stored tips.
    #[must_use]
    pub fn stats(&self) -> TipBankStats {
        let mut categories: BTreeMap<String, usize> = BTreeMap::new();
        let mut total_weight = 0.0;
        for tip in &self.tips {
            *categories
                .entry(tip.category.as_str().to_owned())
                .or_insert(0) += 1;
            total_weight += tip.weight;
        }
        let mean_weight = if self.tips.is_empty() {
            0.0
        } else {
            total_weight / self.tips.len() as f64
        };
        TipBankStats {
            count: self.tips.len(),
            mean_weight,
            categories,
        }
    }

    /// Checks the bank invariants: a usable capacity, a bounded threshold, valid
    /// tips and unique ids.
    pub fn validate(&self) -> Result<()> {
        if self.capacity == 0 {
            return Err(Error::validation(
                "ttse bank `capacity` must be greater than 0",
            ));
        }
        if !self.dedup_threshold.is_finite() || !(0.0..=1.0).contains(&self.dedup_threshold) {
            return Err(Error::validation(format!(
                "ttse bank `dedup_threshold` must be finite and within 0.0..=1.0, got {}",
                self.dedup_threshold
            )));
        }
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for tip in &self.tips {
            tip.validate()?;
            if !seen.insert(tip.tip_id.as_str()) {
                return Err(Error::validation(format!(
                    "ttse bank holds duplicate tip id `{}`",
                    tip.tip_id
                )));
            }
        }
        Ok(())
    }

    /// Merges a cluster of tips into one representative, removing the absorbed
    /// records and returning the representative id.
    ///
    /// The representative is the member with the highest [`TipRecord::score`],
    /// ties going to the lowest `tip_id`. The representative keeps its rule and
    /// category, and absorbs the union of keywords, the maximum weight, the
    /// summed counters, the earliest creation tick and the current tick as
    /// `updated_tick`. Used by the dream pass of a later wave and by the tests
    /// here, so it is crate-visible before its caller exists.
    #[allow(dead_code)]
    pub(crate) fn merge_cluster(&mut self, ids: &[String]) -> Option<String> {
        let wanted: BTreeSet<&str> = ids.iter().map(String::as_str).collect();
        let mut members: Vec<usize> = self
            .tips
            .iter()
            .enumerate()
            .filter(|(_, tip)| wanted.contains(tip.tip_id.as_str()))
            .map(|(index, _)| index)
            .collect();
        if members.is_empty() {
            return None;
        }
        members.sort_by(|left, right| {
            let left = &self.tips[*left];
            let right = &self.tips[*right];
            right
                .score()
                .total_cmp(&left.score())
                .then_with(|| left.tip_id.cmp(&right.tip_id))
        });

        let representative = members[0];
        let mut keywords: Vec<String> = Vec::new();
        let mut weight = self.tips[representative].weight;
        let mut hits = 0u64;
        let mut successes = 0u64;
        let mut failures = 0u64;
        let mut created_tick = self.tips[representative].created_tick;
        let mut absorbed: Vec<String> = Vec::new();
        for index in &members {
            let tip = &self.tips[*index];
            keywords.extend(tip.keywords.iter().cloned());
            weight = weight.max(tip.weight);
            hits = hits.saturating_add(tip.hits);
            successes = successes.saturating_add(tip.successes);
            failures = failures.saturating_add(tip.failures);
            created_tick = created_tick.min(tip.created_tick);
            if *index != representative {
                absorbed.push(tip.tip_id.clone());
            }
        }

        let tick = self.tick;
        let record = &mut self.tips[representative];
        record.keywords = normalize_keywords(keywords);
        record.weight = weight;
        record.hits = hits;
        record.successes = successes;
        record.failures = failures;
        record.created_tick = created_tick;
        record.updated_tick = tick;
        let representative_id = record.tip_id.clone();
        self.tips.retain(|tip| !absorbed.contains(&tip.tip_id));
        Some(representative_id)
    }

    /// Evicts the lowest `(score(), hits, tip_id)` records until the bank fits.
    fn enforce_capacity(&mut self) {
        while self.tips.len() > self.capacity {
            let victim = self
                .tips
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    left.score()
                        .total_cmp(&right.score())
                        .then_with(|| left.hits.cmp(&right.hits))
                        .then_with(|| left.tip_id.cmp(&right.tip_id))
                })
                .map(|(index, _)| index);
            match victim {
                Some(index) => {
                    self.tips.remove(index);
                }
                None => break,
            }
        }
    }
}

/// Merges `incoming` into `existing`, keeping the existing id.
///
/// Representative rule: the longer one, ties broken lexicographically. Category:
/// the higher-scoring record's, ties going to the existing one.
fn merge_record(existing: &mut TipRecord, incoming: &TipRecord, tick: u64) {
    let category = if incoming.score() > existing.score() {
        incoming.category
    } else {
        existing.category
    };
    let rule = match incoming
        .rule
        .chars()
        .count()
        .cmp(&existing.rule.chars().count())
    {
        Ordering::Greater => incoming.rule.clone(),
        Ordering::Less => existing.rule.clone(),
        Ordering::Equal => {
            if incoming.rule <= existing.rule {
                incoming.rule.clone()
            } else {
                existing.rule.clone()
            }
        }
    };
    let keywords = normalize_keywords(
        existing
            .keywords
            .iter()
            .chain(incoming.keywords.iter())
            .cloned(),
    );
    let source = if existing.source.is_empty() {
        incoming.source.clone()
    } else {
        existing.source.clone()
    };
    existing.rule = rule;
    existing.category = category;
    existing.keywords = keywords;
    existing.weight = existing.weight.max(incoming.weight);
    existing.hits = existing.hits.saturating_add(incoming.hits);
    existing.successes = existing.successes.saturating_add(incoming.successes);
    existing.failures = existing.failures.saturating_add(incoming.failures);
    existing.source = source;
    existing.created_tick = existing.created_tick.min(incoming.created_tick);
    existing.updated_tick = tick;
}

/// Aggregate view of a bank, for diagnostics and status lines.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TipBankStats {
    /// The number of stored tips.
    pub count: usize,
    /// The mean weight; `0.0` for an empty bank.
    pub mean_weight: f64,
    /// Tip counts keyed by [`TipCategory::as_str`].
    pub categories: BTreeMap<String, usize>,
}

/// Splits text into deterministic, sorted, deduped tokens.
///
/// ASCII-alphanumeric runs become one lowercase token each; every CJK ideograph
/// or kana character becomes its own token (no dictionary, no stopword list);
/// everything else separates tokens.
#[must_use]
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if is_cjk(character) {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            tokens.push(character.to_string());
        } else if character.is_ascii_alphanumeric() {
            current.push(character.to_ascii_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

/// Whether `character` is a CJK ideograph or kana character.
fn is_cjk(character: char) -> bool {
    matches!(
        character,
        '\u{3040}'..='\u{30FF}' | '\u{3400}'..='\u{4DBF}' | '\u{4E00}'..='\u{9FFF}' | '\u{F900}'..='\u{FAFF}'
    )
}

/// The Jaccard similarity of two token multisets: `|intersection| / |union|`.
///
/// Returns `0.0` when the union is empty and `1.0` for identical non-empty sets.
#[must_use]
pub(crate) fn jaccard(left: &[String], right: &[String]) -> f64 {
    let left: BTreeSet<&String> = left.iter().collect();
    let right: BTreeSet<&String> = right.iter().collect();
    let union = left.union(&right).count();
    if union == 0 {
        return 0.0;
    }
    let intersection = left.intersection(&right).count();
    intersection as f64 / union as f64
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn tip(rule: &str, category: TipCategory) -> TipRecord {
        TipRecord::new(rule, category).expect("valid rule")
    }

    #[test]
    fn add_merges_above_threshold_and_inserts_below() {
        let first = tip("run cargo fmt before committing", TipCategory::Workflow)
            .with_keywords(vec!["fmt".to_owned()]);
        // Five of six tokens shared: 5 / 6 ~= 0.83.
        let similar = tip("run cargo fmt before committing now", TipCategory::Workflow)
            .with_observation(2, 1, 1);

        let mut bank = TipBank::new(16, 0.8);
        assert_eq!(
            bank.add(first.clone()),
            AddOutcome::Inserted(first.tip_id.clone())
        );
        match bank.add(similar.clone()) {
            AddOutcome::Merged { into, similarity } => {
                assert_eq!(into, first.tip_id);
                assert!(
                    similarity >= 0.8,
                    "similarity {similarity} must reach the threshold"
                );
            }
            other @ AddOutcome::Inserted(_) => panic!("expected a merge, got {other:?}"),
        }
        assert_eq!(bank.len(), 1);
        let merged = bank.get(&first.tip_id).expect("representative kept");
        assert_eq!(merged.rule, "run cargo fmt before committing now");
        assert_eq!((merged.hits, merged.successes, merged.failures), (2, 1, 1));
        assert_eq!(merged.keywords, vec!["fmt".to_owned()]);

        let unrelated = bank.add(tip("check the workspace path", TipCategory::Environment));
        assert!(matches!(unrelated, AddOutcome::Inserted(_)));
        assert_eq!(bank.len(), 2);
        assert_eq!(unrelated.tip_id(), bank.tips()[1].tip_id.as_str());
        assert!(bank.validate().is_ok());

        let mut strict = TipBank::new(16, 0.9);
        strict.add(first);
        assert!(matches!(strict.add(similar), AddOutcome::Inserted(_)));
        assert_eq!(strict.len(), 2);
    }

    #[test]
    fn add_stamps_ticks_and_observation_tracks_success_rate() {
        let mut bank = TipBank::new(4, 0.99);
        assert_eq!(bank.tick(), 0);
        assert_eq!(bank.increment_tick(), 1);
        let tip = tip("use rg instead of grep", TipCategory::Tool);
        bank.add(tip.clone());
        let stored = bank.get(&tip.tip_id).expect("stored");
        assert_eq!((stored.created_tick, stored.updated_tick), (1, 1));

        assert!(bank.observe(&tip.tip_id, true));
        assert_eq!(bank.increment_tick(), 2);
        assert!(bank.observe(&tip.tip_id, false));
        assert!(!bank.observe("missing", true));
        let stored = bank.get(&tip.tip_id).expect("stored");
        assert_eq!((stored.hits, stored.successes, stored.failures), (2, 1, 1));
        assert_eq!(stored.updated_tick, 2);
        assert!((stored.success_rate() - 0.5).abs() < f64::EPSILON);

        assert_eq!(
            bank.remove(&tip.tip_id).map(|tip| tip.tip_id),
            Some(tip.tip_id.clone())
        );
        assert!(bank.is_empty());
        assert!(bank.remove(&tip.tip_id).is_none());
    }

    #[test]
    fn top_ranks_by_score_then_id_and_boosts_on_query() {
        let mut bank = TipBank::new(8, 0.99);
        let fmt = tip("run cargo fmt", TipCategory::Tool);
        let env = tip("check environment variables", TipCategory::Environment);
        bank.add(fmt.clone());
        bank.add(env.clone());

        let ranked = bank.top(8, None);
        assert_eq!(ranked.len(), 2);
        assert!(ranked[0].tip_id < ranked[1].tip_id);

        let ranked = bank.top(8, Some("environment variables"));
        assert_eq!(ranked[0].tip_id, env.tip_id);
        assert_eq!(ranked[1].tip_id, fmt.tip_id);
        assert_eq!(bank.top(1, None).len(), 1);
    }

    #[test]
    fn capacity_eviction_drops_the_lowest_ranked_tip() {
        let mut bank = TipBank::with_capacity(2);
        let mut ids: Vec<String> = Vec::new();
        for rule in [
            "run the formatter first",
            "inspect the workspace layout",
            "ask before deleting files",
        ] {
            let tip = tip(rule, TipCategory::Other);
            ids.push(tip.tip_id.clone());
            bank.add(tip);
        }
        ids.sort_unstable();
        assert_eq!(bank.len(), 2);
        let mut surviving: Vec<String> = bank.tips().iter().map(|tip| tip.tip_id.clone()).collect();
        surviving.sort_unstable();
        assert_eq!(surviving, ids[1..].to_vec());
        assert!(bank.get(&ids[0]).is_none());
    }

    #[test]
    fn prune_decay_and_stats() {
        let config = TtseConfig {
            dream_ttl_ticks: 10,
            ..TtseConfig::default()
        };
        let mut bank = TipBank::new(8, 0.99);
        let stale = tip("stale rule about nothing", TipCategory::Other);
        let light = tip("light rule about nothing at all", TipCategory::Other)
            .with_weight(0.01)
            .expect("positive weight");
        let fresh = tip("keep this rule around", TipCategory::Tool);
        bank.add(stale.clone());
        bank.add(light.clone());
        bank.add(fresh.clone());
        for _ in 0..11 {
            bank.increment_tick();
        }
        bank.observe(&fresh.tip_id, true);

        assert_eq!(bank.prune(&config), 2);
        assert_eq!(bank.len(), 1);
        assert_eq!(bank.tips()[0].tip_id, fresh.tip_id);

        let stats = bank.stats();
        assert_eq!(stats.count, 1);
        assert_eq!(stats.categories.get("tool"), Some(&1));
        assert!((stats.mean_weight - 1.0).abs() < f64::EPSILON);
        assert_eq!(TipBank::default().stats(), TipBankStats::default());

        bank.decay(0.5);
        assert!((bank.tips()[0].weight - 0.5).abs() < f64::EPSILON);
        bank.decay(f64::NAN);
        bank.decay(-1.0);
        assert!((bank.tips()[0].weight - 0.5).abs() < f64::EPSILON);
        bank.decay(0.0);
        assert!((bank.tips()[0].weight).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_counts_categories() {
        let mut bank = TipBank::new(8, 0.99);
        bank.add(tip("run cargo test", TipCategory::Tool));
        bank.add(tip("check the workspace", TipCategory::Environment));
        bank.add(tip("never skip the dry run", TipCategory::Pitfall));
        let stats = bank.stats();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.categories.get("tool"), Some(&1));
        assert_eq!(stats.categories.get("environment"), Some(&1));
        assert_eq!(stats.categories.get("pitfall"), Some(&1));
        assert_eq!(bank.by_category(TipCategory::Tool).len(), 1);
        assert_eq!(bank.by_category(TipCategory::Workflow).len(), 0);
    }

    #[test]
    fn tokenize_handles_ascii_and_cjk() {
        assert_eq!(
            tokenize("Cargo FMT, cargo"),
            vec!["cargo".to_owned(), "fmt".to_owned()]
        );
        assert!(tokenize("运行命令")
            .iter()
            .all(|token| token.chars().count() == 1));
        assert_eq!(tokenize("运行命令").len(), 4);
        assert_eq!(tokenize("   ").len(), 0);
        assert!(
            (jaccard(&tokenize("run cargo fmt"), &tokenize("run cargo fmt")) - 1.0).abs()
                < f64::EPSILON
        );
        assert!((jaccard(&tokenize("run cargo"), &tokenize("check env"))).abs() < f64::EPSILON);
        assert!(jaccard(&[], &[]).abs() < f64::EPSILON);
        assert!((jaccard(&tokenize("a b"), &tokenize("b c")) - 1.0 / 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn merge_cluster_absorbs_members() {
        let mut bank = TipBank::new(8, 0.99);
        let weak = tip("weak rule", TipCategory::Tool).with_observation(1, 0, 1);
        let strong = tip("strong rule", TipCategory::Pitfall).with_observation(1, 1, 0);
        bank.add(weak.clone());
        bank.add(strong.clone());
        assert_eq!(bank.increment_tick(), 1);

        let merged = bank
            .merge_cluster(&[weak.tip_id.clone(), strong.tip_id.clone()])
            .expect("cluster merged");
        assert_eq!(merged, strong.tip_id, "higher score wins");
        assert_eq!(bank.len(), 1);
        let record = bank.get(&merged).expect("representative kept");
        assert_eq!(record.rule, "strong rule");
        assert_eq!(record.category, TipCategory::Pitfall);
        assert_eq!(record.hits, 2);
        assert_eq!((record.successes, record.failures), (1, 1));
        assert_eq!(record.updated_tick, 1);
        assert!(bank.merge_cluster(&[]).is_none());
        assert!(bank.merge_cluster(&["missing".to_owned()]).is_none());
    }

    #[test]
    fn validate_rejects_duplicate_ids_and_bad_thresholds() {
        let mut bank = TipBank::new(4, 0.5);
        let tip = tip("unique rule", TipCategory::Tool);
        bank.add(tip.clone());
        assert!(bank.validate().is_ok());

        bank.tips.push(tip);
        assert!(bank.validate().is_err());

        let broken = TipBank {
            dedup_threshold: 2.0,
            ..TipBank::default()
        };
        assert!(broken.validate().is_err());
        let broken = TipBank {
            capacity: 0,
            ..TipBank::default()
        };
        assert!(broken.validate().is_err());
    }

    #[test]
    fn serde_round_trip_keeps_the_python_shape() {
        let mut bank = TipBank::new(3, 0.75);
        bank.add(tip("run the tests offline", TipCategory::Workflow));
        bank.increment_tick();
        let value = serde_json::to_value(&bank).expect("serializable");
        assert_eq!(value["capacity"], json!(3));
        assert_eq!(value["tick"], json!(1));
        assert!(value["tips"].is_array());
        assert!(value["dedup_threshold"].is_number());

        let restored: TipBank = serde_json::from_value(value).expect("deserializable");
        assert_eq!(restored, bank);

        let partial: TipBank =
            serde_json::from_value(json!({ "tips": [], "future": 1 })).expect("partial bank");
        assert_eq!(partial.capacity, DEFAULT_CAPACITY);
        assert!((partial.dedup_threshold - DEFAULT_DEDUP_THRESHOLD).abs() < f64::EPSILON);
    }

    #[test]
    fn new_normalizes_capacity_and_threshold() {
        let bank = TipBank::new(0, 5.0);
        assert_eq!(bank.capacity, 1);
        assert!((bank.dedup_threshold - 1.0).abs() < f64::EPSILON);
        let bank = TipBank::new(0, f64::NAN);
        assert!((bank.dedup_threshold - DEFAULT_DEDUP_THRESHOLD).abs() < f64::EPSILON);
        assert_eq!(TipBank::with_capacity(4).capacity, 4);
        assert_eq!(TipBank::default().capacity, DEFAULT_CAPACITY);
    }
}
