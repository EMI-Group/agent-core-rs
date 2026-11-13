//! Offline "dreaming": clustering, merging and pruning of the tip bank.
//!
//! Ports the deterministic part of Python `openjiuwen.agent_evolving.ttse.dream`:
//! near-duplicate tips are clustered by keyword overlap and collapsed onto a
//! representative, then stale or negligible tips are pruned. No model call is
//! involved in the offline path.
//!
//! Clustering is connected-component analysis over the pairwise `jaccard`
//! similarity of tip rules, so the pass is order-independent and idempotent:
//! once a component has been collapsed onto one representative, no pair of
//! surviving tips can still reach the merge threshold.

use crate::ttse::bank::{jaccard, tokenize, TipBank};
use crate::ttse::config::TtseConfig;
use crate::ttse::models::DreamReport;

/// The smallest cluster the dreamer will ever merge.
///
/// A cluster of one is a no-op, so [`TtseConfig::dream_cluster_min_size`] values
/// below `2` are raised to this floor instead of merging nothing or everything.
const MIN_CLUSTER_SIZE: usize = 2;

/// Clusters, merges and prunes the tips of a [`TipBank`](super::TipBank).
#[derive(Debug, Clone)]
pub struct Dreamer {
    config: TtseConfig,
}

impl Dreamer {
    /// Builds a dreamer from the shared TTSE configuration.
    #[must_use]
    pub fn new(config: TtseConfig) -> Self {
        Self { config }
    }

    /// The configuration this dreamer applies.
    #[must_use]
    pub fn config(&self) -> &TtseConfig {
        &self.config
    }

    /// Whether the bank is due for a dream pass.
    ///
    /// A pass is due when dreaming is enabled, the bank holds at least
    /// [`TtseConfig::dream_min_rules`] tips and the bank tick is a positive
    /// multiple of [`TtseConfig::dream_interval`].
    #[must_use]
    pub fn should_dream(&self, bank: &TipBank) -> bool {
        let interval = self.config.dream_interval;
        let tick = bank.tick();
        self.config.dream_enabled
            && bank.len() >= self.config.dream_min_rules
            && tick > 0
            && interval > 0
            && tick % interval == 0
    }

    /// Runs one dream pass over `bank` and reports what changed.
    ///
    /// The pass has three deterministic stages:
    ///
    /// 1. Every tip id is collected in ascending order and tips joined by a
    ///    similarity edge are grouped into connected clusters. Two tips are
    ///    joined when the Jaccard similarity of their rule tokens reaches
    ///    [`TtseConfig::dream_merge_threshold`] (clamped to `0.0..=1.0`, a
    ///    non-finite threshold meaning exactly equal rules).
    /// 2. Clusters are processed in ascending order of their smallest member id.
    ///    A cluster of at least [`TtseConfig::dream_cluster_min_size`] tips is
    ///    collapsed onto the best-scoring member by
    ///    `TipBank::merge_cluster`; every absorbed tip counts towards
    ///    [`DreamReport::merged`] and the surviving representative towards
    ///    [`DreamReport::promoted`]. Singleton clusters are left untouched.
    /// 3. Stale and negligible tips are pruned, but only when
    ///    [`TtseConfig::dream_prune_enabled`] is set.
    pub fn dream(&mut self, bank: &mut TipBank) -> DreamReport {
        let examined = bank.len();
        let threshold = merge_threshold(self.config.dream_merge_threshold);

        let mut ids: Vec<String> = bank.tips.iter().map(|tip| tip.tip_id.clone()).collect();
        ids.sort_unstable();

        let token_sets: Vec<Vec<String>> = ids
            .iter()
            .map(|tip_id| {
                let rule = bank.get(tip_id).map_or("", |tip| tip.rule.as_str());
                tokenize(rule)
            })
            .collect();

        let mut clusters = connected_clusters(&ids, &token_sets, threshold);
        clusters.sort_by(|left, right| left.first().cmp(&right.first()));

        let mut report = DreamReport {
            examined,
            ..DreamReport::default()
        };
        let min_size = self.config.dream_cluster_min_size.max(MIN_CLUSTER_SIZE);
        for cluster in &clusters {
            if cluster.len() < min_size {
                continue;
            }
            let absorbed = cluster.len() - 1;
            if bank.merge_cluster(cluster).is_some() {
                report.merged += absorbed;
                report.promoted += 1;
            }
        }

        if self.config.dream_prune_enabled {
            report.pruned = bank.prune(&self.config);
        }
        report
    }
}

/// Clamps a configured merge threshold into `0.0..=1.0`.
///
/// A non-finite threshold collapses to `1.0`, which only merges rules with
/// identical token sets, instead of poisoning every comparison.
fn merge_threshold(threshold: f64) -> f64 {
    if threshold.is_finite() {
        threshold.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Groups `ids` into connected components using the matching `token_sets`.
///
/// `ids` and `token_sets` describe the same tips. `ids` must be sorted
/// ascending; the returned clusters then list their tip ids ascending and are
/// themselves ordered by their first member.
fn connected_clusters(
    ids: &[String],
    token_sets: &[Vec<String>],
    threshold: f64,
) -> Vec<Vec<String>> {
    let mut parents: Vec<usize> = (0..token_sets.len()).collect();
    for (left, left_tokens) in token_sets.iter().enumerate() {
        for (right, right_tokens) in token_sets.iter().enumerate().skip(left + 1) {
            if jaccard(left_tokens, right_tokens) < threshold {
                continue;
            }
            let left_root = find(&mut parents, left);
            let right_root = find(&mut parents, right);
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
    for (index, member) in ids.iter().enumerate() {
        let root = find(&mut parents, index);
        if let Some(cluster_index) = cluster_of_root[root] {
            clusters[cluster_index].push(member.clone());
        } else {
            cluster_of_root[root] = Some(clusters.len());
            clusters.push(vec![member.clone()]);
        }
    }
    clusters
}

/// The root of `node` in the disjoint-set forest, compressing the path.
fn find(parents: &mut [usize], node: usize) -> usize {
    let mut root = node;
    while parents[root] != root {
        root = parents[root];
    }
    let mut current = node;
    while parents[current] != root {
        let next = parents[current];
        parents[current] = root;
        current = next;
    }
    root
}

#[cfg(test)]
mod tests {
    use crate::ttse::models::{TipCategory, TipRecord};

    use super::*;

    /// Three rules whose token sets overlap well above the default threshold.
    const DUPLICATES: [&str; 3] = [
        "always check the workspace path before running tests",
        "always check the workspace path before running the tests",
        "always check the workspace path before running all tests",
    ];
    /// A rule sharing no token with [`DUPLICATES`].
    const UNRELATED: &str = "prefer rg over grep for speed";

    fn bank_with(rules: &[&str], tick: u64) -> TipBank {
        let mut bank = TipBank::with_capacity(16);
        bank.tick = tick;
        for rule in rules {
            bank.tips.push(
                TipRecord::new(*rule, TipCategory::Tool).expect("the test rules are not blank"),
            );
        }
        bank
    }

    fn pass(config: TtseConfig) -> Dreamer {
        Dreamer::new(config)
    }

    #[test]
    fn dream_merges_near_duplicates_onto_one_representative() {
        let config = TtseConfig {
            dream_prune_enabled: false,
            ..TtseConfig::default()
        };
        let mut dreamer = pass(config);
        let mut bank = bank_with(&DUPLICATES, 20);

        let report = dreamer.dream(&mut bank);

        assert_eq!(report.examined, 3);
        assert_eq!(report.merged, 2);
        assert_eq!(report.promoted, 1);
        assert_eq!(report.pruned, 0);
        assert_eq!(bank.len(), 1);
        assert_eq!(bank.tips()[0].updated_tick, 20);
        assert_eq!(dreamer.config().dream_cluster_min_size, 2);
    }

    #[test]
    fn dissimilar_tips_survive_dreaming() {
        let config = TtseConfig {
            dream_prune_enabled: false,
            ..TtseConfig::default()
        };
        let mut dreamer = pass(config);
        let mut rules: Vec<&str> = DUPLICATES.to_vec();
        rules.push(UNRELATED);
        let mut bank = bank_with(&rules, 20);

        let report = dreamer.dream(&mut bank);

        assert_eq!(report.examined, 4);
        assert_eq!(report.merged, 2);
        assert_eq!(report.promoted, 1);
        assert_eq!(bank.len(), 2);
        let survivor = bank
            .tips()
            .iter()
            .find(|tip| tip.rule == UNRELATED)
            .expect("the dissimilar tip is kept");
        assert_eq!(survivor.hits, 0);
        assert_eq!(survivor.updated_tick, 0, "an untouched tip keeps its tick");
    }

    #[test]
    fn cluster_min_size_gates_the_merge() {
        let config = TtseConfig {
            dream_prune_enabled: false,
            dream_cluster_min_size: 3,
            ..TtseConfig::default()
        };
        let mut dreamer = pass(config);
        let mut small = bank_with(&DUPLICATES[..2], 20);
        let report = dreamer.dream(&mut small);
        assert_eq!(report.examined, 2);
        assert_eq!((report.merged, report.promoted), (0, 0));
        assert_eq!(small.len(), 2);

        let mut large = bank_with(&DUPLICATES, 20);
        let report = dreamer.dream(&mut large);
        assert_eq!(report.examined, 3);
        assert_eq!((report.merged, report.promoted), (2, 1));
        assert_eq!(large.len(), 1);
    }

    #[test]
    fn should_dream_requires_every_gate() {
        let dreamer = pass(TtseConfig {
            dream_min_rules: 8,
            ..TtseConfig::default()
        });
        let mut bank = TipBank::with_capacity(16);
        assert!(!dreamer.should_dream(&bank), "an empty bank at tick 0");
        bank.tick = 20;
        assert!(
            !dreamer.should_dream(&bank),
            "fewer tips than dream_min_rules"
        );

        for index in 0..8 {
            bank.tips.push(
                TipRecord::new(format!("distinct rule {index}"), TipCategory::Other)
                    .expect("the test rules are not blank"),
            );
        }
        assert!(dreamer.should_dream(&bank), "tick 20 is a multiple of 20");

        bank.tick = 21;
        assert!(!dreamer.should_dream(&bank), "tick 21 is not a multiple");
        bank.tick = 20;

        let disabled = Dreamer::new(TtseConfig {
            dream_enabled: false,
            dream_min_rules: 8,
            ..TtseConfig::default()
        });
        assert!(!disabled.should_dream(&bank));

        let no_interval = Dreamer::new(TtseConfig {
            dream_interval: 0,
            dream_min_rules: 8,
            ..TtseConfig::default()
        });
        assert!(!no_interval.should_dream(&bank));

        let larger_floor = Dreamer::new(TtseConfig {
            dream_min_rules: 9,
            ..TtseConfig::default()
        });
        assert!(!larger_floor.should_dream(&bank));
    }

    #[test]
    fn pruning_follows_the_config_switch() {
        let stale = [UNRELATED];
        let enabled = TtseConfig {
            dream_ttl_ticks: 10,
            ..TtseConfig::default()
        };
        let mut dreamer = pass(enabled);
        let mut bank = bank_with(&stale, 100);
        let report = dreamer.dream(&mut bank);
        assert_eq!(report.examined, 1);
        assert_eq!(report.pruned, 1);
        assert!(bank.is_empty());

        let disabled = TtseConfig {
            dream_ttl_ticks: 10,
            dream_prune_enabled: false,
            ..TtseConfig::default()
        };
        let mut dreamer = pass(disabled);
        let mut bank = bank_with(&stale, 100);
        let report = dreamer.dream(&mut bank);
        assert_eq!(report.examined, 1);
        assert_eq!(report.pruned, 0);
        assert_eq!(bank.len(), 1);
    }

    #[test]
    fn a_second_dream_is_idempotent() {
        let config = TtseConfig {
            dream_prune_enabled: false,
            ..TtseConfig::default()
        };
        let mut dreamer = pass(config);
        let mut rules: Vec<&str> = DUPLICATES.to_vec();
        rules.push(UNRELATED);
        let mut bank = bank_with(&rules, 20);

        let first = dreamer.dream(&mut bank);
        assert_eq!((first.merged, first.promoted), (2, 1));
        assert_eq!(bank.len(), 2);

        let second = dreamer.dream(&mut bank);
        assert_eq!(second.examined, 2);
        assert_eq!((second.merged, second.promoted), (0, 0));
        assert_eq!(bank.len(), 2);
        assert!(bank.validate().is_ok());
    }
}
