//! Rollout classifiers, stop validators, samplers and their registry
//! (Python `coordinator/processors.py`).
//!
//! Python discovers the processors by reflection (`inspect.getmembers` over the
//! module). Rust has no runtime reflection, so [`ProcessorsRegistry`] registers
//! each built-in under its Python name explicitly and rejects an unknown name
//! with [`Error::validation`] listing what *is* registered — Python's
//! `AGENT_RL_PROCESSOR_NOT_FOUND` with the same "available" payload.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};

use crate::rl::RolloutWithReward;

/// The name of the built-in classifier (`default_classify_rollouts`).
pub const DEFAULT_CLASSIFIER: &str = "default_classify_rollouts";
/// The name of the built-in stop validator (`default_validate_stop`).
pub const DEFAULT_VALIDATOR: &str = "default_validate_stop";
/// The name of the balanced stop validator (`validate_stop_balanced`).
pub const BALANCED_VALIDATOR: &str = "validate_stop_balanced";
/// The name of the identity sampler (`default_sampling`).
pub const DEFAULT_SAMPLER: &str = "default_sampling";
/// The name of the balanced adaptive sampler (`sampling_ada`).
pub const ADAPTIVE_SAMPLER: &str = "sampling_ada";

/// The positive and negative halves of a classified rollout list.
pub type ClassifiedRollouts = (Vec<RolloutWithReward>, Vec<RolloutWithReward>);

/// Rollouts grouped by prompt (`origin_task_id`).
pub type RolloutCache = BTreeMap<String, Vec<RolloutWithReward>>;

/// The positive and negative caches returned by a sampler.
pub type SampledCaches = (RolloutCache, RolloutCache);

/// A rollout classifier: one list in, positive/negative lists out.
pub type ClassifierFn = dyn Fn(&[RolloutWithReward]) -> ClassifiedRollouts + Send + Sync;

/// A stop validator: the caches so far plus the optional per-prompt keep target.
///
/// Python binds `final_keep_per_prompt` with `functools.partial` when the
/// function accepts it. Rust has no signature inspection, so the target is
/// always passed and a validator that does not need it ignores the argument.
pub type ValidatorFn =
    dyn Fn(&[RolloutWithReward], &[RolloutWithReward], Option<usize>) -> bool + Send + Sync;

/// A sampler: the positive/negative caches plus the optional per-prompt target.
pub type SamplerFn =
    dyn Fn(&RolloutCache, &RolloutCache, Option<usize>) -> SampledCaches + Send + Sync;

/// Splits rollouts into positives (`reward > 0`) and negatives.
#[must_use]
pub fn classify_rollouts(mdp_list: &[RolloutWithReward]) -> ClassifiedRollouts {
    let mut positive = Vec::new();
    let mut negative = Vec::new();
    for rollout in mdp_list {
        // A non-finite reward cannot be judged positive: it would poison the
        // reward statistics and never equal 1.0 anyway.
        if rollout.reward > 0.0 {
            positive.push(rollout.clone());
        } else {
            negative.push(rollout.clone());
        }
    }
    (positive, negative)
}

/// The default stop rule.
///
/// Never stops while negatives outnumber positives or fewer than two positives
/// exist; otherwise stops as soon as one positive reward is exactly `1.0`.
#[must_use]
pub fn validate_stop(
    pos_rollout_list: &[RolloutWithReward],
    neg_rollout_list: &[RolloutWithReward],
) -> bool {
    if neg_rollout_list.len() > pos_rollout_list.len() {
        return false;
    }
    if pos_rollout_list.len() < 2 {
        return false;
    }
    pos_rollout_list
        .iter()
        .any(|rollout| rollout.reward.partial_cmp(&1.0) == Some(std::cmp::Ordering::Equal))
}

/// The balanced stop rule.
///
/// Stops once both halves reached their share of `final_keep_per_prompt`
/// (positives take the smaller half on an odd target).
#[must_use]
pub fn validate_stop_balanced(
    pos_rollout_list: &[RolloutWithReward],
    neg_rollout_list: &[RolloutWithReward],
    final_keep_per_prompt: usize,
) -> bool {
    let target_pos = final_keep_per_prompt / 2;
    let target_neg = final_keep_per_prompt - target_pos;
    pos_rollout_list.len() >= target_pos && neg_rollout_list.len() >= target_neg
}

/// The identity sampler: returns deep copies of both caches.
#[must_use]
pub fn sampling(pos_rollout_dict: &RolloutCache, neg_rollout_dict: &RolloutCache) -> SampledCaches {
    (pos_rollout_dict.clone(), neg_rollout_dict.clone())
}

/// Downsamples one prompt's rollouts toward a balanced `target_total`.
///
/// Each half takes at most half the target first, then the spare capacity of
/// the other half fills the gap, so a lopsided prompt still reaches the target
/// when the total allows it. Selection keeps the first entries, so the result
/// depends only on input order.
#[must_use]
pub fn downsample_one_uid(
    pos_list: &[RolloutWithReward],
    neg_list: &[RolloutWithReward],
    target_total: usize,
) -> ClassifiedRollouts {
    let mut target_pos = (target_total / 2).min(pos_list.len());
    let mut target_neg = target_total.saturating_sub(target_pos).min(neg_list.len());

    let taken = target_pos + target_neg;
    if taken < target_total {
        let mut remaining = target_total - taken;
        let extra_pos = pos_list.len() - target_pos;
        if extra_pos > 0 {
            let add = extra_pos.min(remaining);
            target_pos += add;
            remaining -= add;
        }
        if remaining > 0 {
            let extra_neg = neg_list.len() - target_neg;
            if extra_neg > 0 {
                target_neg += extra_neg.min(remaining);
            }
        }
    }

    (
        pos_list[..target_pos].to_vec(),
        neg_list[..target_neg].to_vec(),
    )
}

/// The balanced adaptive sampler: [`downsample_one_uid`] for every prompt.
///
/// Prompts present in only one cache are still sampled (their empty half stays
/// empty), and prompts absent from both are skipped. The output is keyed by the
/// union of the input keys, so iteration order is deterministic.
#[must_use]
pub fn sampling_ada(
    pos_rollout_dict: &RolloutCache,
    neg_rollout_dict: &RolloutCache,
    final_keep_per_prompt: usize,
) -> SampledCaches {
    let mut out_pos = RolloutCache::new();
    let mut out_neg = RolloutCache::new();
    let mut uids: Vec<&str> = pos_rollout_dict
        .keys()
        .chain(neg_rollout_dict.keys())
        .map(String::as_str)
        .collect();
    uids.sort_unstable();
    uids.dedup();

    for uid in uids {
        let empty = Vec::new();
        let pos_list = pos_rollout_dict.get(uid).unwrap_or(&empty);
        let neg_list = neg_rollout_dict.get(uid).unwrap_or(&empty);
        if pos_list.is_empty() && neg_list.is_empty() {
            continue;
        }
        let (pos_selected, neg_selected) =
            downsample_one_uid(pos_list, neg_list, final_keep_per_prompt);
        out_pos.insert(uid.to_owned(), pos_selected);
        out_neg.insert(uid.to_owned(), neg_selected);
    }
    (out_pos, out_neg)
}

/// A registry of classifiers, validators and samplers, keyed by name.
///
/// [`ProcessorsRegistry::new`] pre-registers every built-in under its Python
/// name. Registration is a plain map insert: it replaces a previous entry and
/// returns it, so custom processors can shadow a built-in deliberately.
#[derive(Clone, Default)]
pub struct ProcessorsRegistry {
    classifiers: BTreeMap<String, Arc<ClassifierFn>>,
    validators: BTreeMap<String, Arc<ValidatorFn>>,
    samplers: BTreeMap<String, Arc<SamplerFn>>,
}

impl fmt::Debug for ProcessorsRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProcessorsRegistry")
            .field("classifiers", &self.classifier_names())
            .field("validators", &self.validator_names())
            .field("samplers", &self.sampler_names())
            .finish()
    }
}

impl ProcessorsRegistry {
    /// Creates a registry with the built-in processors registered.
    #[must_use]
    pub fn new() -> Self {
        let mut registry = Self::default();
        registry.register_classifier(DEFAULT_CLASSIFIER, |rollouts| classify_rollouts(rollouts));
        registry.register_validator(DEFAULT_VALIDATOR, |pos, neg, _keep| validate_stop(pos, neg));
        registry.register_validator(BALANCED_VALIDATOR, |pos, neg, keep| {
            validate_stop_balanced(pos, neg, keep.unwrap_or(8))
        });
        registry.register_sampler(DEFAULT_SAMPLER, |pos, neg, _keep| sampling(pos, neg));
        registry.register_sampler(ADAPTIVE_SAMPLER, |pos, neg, keep| {
            sampling_ada(pos, neg, keep.unwrap_or(8))
        });
        registry
    }

    /// Registers a classifier under `name`, returning the shadowed one.
    pub fn register_classifier<F>(
        &mut self,
        name: impl Into<String>,
        classifier: F,
    ) -> Option<Arc<ClassifierFn>>
    where
        F: Fn(&[RolloutWithReward]) -> ClassifiedRollouts + Send + Sync + 'static,
    {
        self.classifiers.insert(name.into(), Arc::new(classifier))
    }

    /// Registers a stop validator under `name`, returning the shadowed one.
    pub fn register_validator<F>(
        &mut self,
        name: impl Into<String>,
        validator: F,
    ) -> Option<Arc<ValidatorFn>>
    where
        F: Fn(&[RolloutWithReward], &[RolloutWithReward], Option<usize>) -> bool
            + Send
            + Sync
            + 'static,
    {
        self.validators.insert(name.into(), Arc::new(validator))
    }

    /// Registers a sampler under `name`, returning the shadowed one.
    pub fn register_sampler<F>(
        &mut self,
        name: impl Into<String>,
        sampler: F,
    ) -> Option<Arc<SamplerFn>>
    where
        F: Fn(&RolloutCache, &RolloutCache, Option<usize>) -> SampledCaches + Send + Sync + 'static,
    {
        self.samplers.insert(name.into(), Arc::new(sampler))
    }

    /// Returns the classifier registered under `name`.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the registered classifiers when `name` is
    /// unknown.
    pub fn classifier(&self, name: &str) -> Result<Arc<ClassifierFn>> {
        self.classifiers
            .get(name)
            .cloned()
            .ok_or_else(|| unknown("classifier", name, self.classifier_names()))
    }

    /// Returns the validator registered under `name`.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the registered validators when `name` is
    /// unknown.
    pub fn validator(&self, name: &str) -> Result<Arc<ValidatorFn>> {
        self.validators
            .get(name)
            .cloned()
            .ok_or_else(|| unknown("validator", name, self.validator_names()))
    }

    /// Returns the sampler registered under `name`.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the registered samplers when `name` is
    /// unknown.
    pub fn sampler(&self, name: &str) -> Result<Arc<SamplerFn>> {
        self.samplers
            .get(name)
            .cloned()
            .ok_or_else(|| unknown("sampler", name, self.sampler_names()))
    }

    /// The registered classifier names, sorted.
    #[must_use]
    pub fn classifier_names(&self) -> Vec<String> {
        self.classifiers.keys().cloned().collect()
    }

    /// The registered validator names, sorted.
    #[must_use]
    pub fn validator_names(&self) -> Vec<String> {
        self.validators.keys().cloned().collect()
    }

    /// The registered sampler names, sorted.
    #[must_use]
    pub fn sampler_names(&self) -> Vec<String> {
        self.samplers.keys().cloned().collect()
    }
}

fn unknown(kind: &str, name: &str, available: Vec<String>) -> Error {
    Error::validation(format!(
        "unknown {kind} {name:?}; registered: {available:?}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rollout(turn: u64, reward: f64) -> RolloutWithReward {
        RolloutWithReward {
            turn_id: turn,
            task_id: "t-1".to_owned(),
            rollout_id: "r-1".to_owned(),
            input_prompt_ids: vec![1],
            output_response_ids: vec![2],
            reward,
            n_turns: 1,
            loss_mask: Vec::new(),
            group_id: Some("g".to_owned()),
        }
    }

    fn cache(entries: &[(&str, &[f64])]) -> RolloutCache {
        entries
            .iter()
            .map(|(uid, rewards)| {
                (
                    (*uid).to_owned(),
                    rewards
                        .iter()
                        .enumerate()
                        .map(|(index, reward)| rollout(index as u64, *reward))
                        .collect(),
                )
            })
            .collect()
    }

    fn rewards(rollouts: &[RolloutWithReward]) -> Vec<f64> {
        rollouts.iter().map(|rollout| rollout.reward).collect()
    }

    #[test]
    fn classification_is_positive_exactly_above_zero() {
        let rollouts = [
            rollout(0, 1.0),
            rollout(1, 0.0),
            rollout(2, -0.5),
            rollout(3, 0.25),
            rollout(4, f64::NAN),
        ];
        let (pos, neg) = classify_rollouts(&rollouts);
        assert_eq!(rewards(&pos), vec![1.0, 0.25]);
        assert_eq!(neg.len(), 3);
        assert!(rewards(&neg).contains(&0.0));
        assert!(classify_rollouts(&[]).0.is_empty(), "no input, no positive");
    }

    #[test]
    fn the_default_validator_needs_two_positives_and_one_perfect_reward() {
        let pos = |values: &[f64]| -> Vec<RolloutWithReward> {
            values
                .iter()
                .enumerate()
                .map(|(index, reward)| rollout(index as u64, *reward))
                .collect()
        };
        // Fewer than two positives.
        assert!(!validate_stop(&pos(&[1.0]), &[]));
        // Two positives, but none perfect.
        assert!(!validate_stop(&pos(&[0.5, 0.5]), &[]));
        // Two positives, one perfect.
        assert!(validate_stop(&pos(&[0.5, 1.0]), &[]));
        // More negatives than positives.
        assert!(!validate_stop(&pos(&[1.0, 1.0]), &pos(&[0.0, 0.0, 0.0])));
        // Equal counts are fine.
        assert!(validate_stop(&pos(&[1.0, 0.5]), &pos(&[0.0, 0.0])));
    }

    #[test]
    fn the_balanced_validator_targets_both_halves() {
        let pos = |count: usize| -> Vec<RolloutWithReward> {
            (0..count).map(|index| rollout(index as u64, 1.0)).collect()
        };
        // Target 8 -> 4 positives and 4 negatives.
        assert!(!validate_stop_balanced(&pos(4), &pos(3), 8));
        assert!(validate_stop_balanced(&pos(4), &pos(4), 8));
        // Odd target: positives take the smaller half.
        assert!(validate_stop_balanced(&pos(2), &pos(3), 5));
        assert!(!validate_stop_balanced(&pos(3), &pos(2), 5));
        // A zero target is satisfied immediately.
        assert!(validate_stop_balanced(&[], &[], 0));
    }

    #[test]
    fn identity_sampling_copies_both_caches() {
        let pos = cache(&[("a", &[1.0])]);
        let neg = cache(&[("a", &[0.0])]);
        let (out_pos, out_neg) = sampling(&pos, &neg);
        assert_eq!(out_pos, pos);
        assert_eq!(out_neg, neg);
        assert_eq!(out_pos["a"][0].reward, 1.0);
    }

    #[test]
    fn downsampling_balances_and_fills_spare_capacity() {
        let pos: Vec<_> = (0..10).map(|i| rollout(i, 1.0)).collect();
        let neg: Vec<_> = (0..10).map(|i| rollout(i, 0.0)).collect();

        let (sel_pos, sel_neg) = downsample_one_uid(&pos, &neg, 8);
        assert_eq!(sel_pos.len(), 4);
        assert_eq!(sel_neg.len(), 4);

        // Only positives: fill the whole target from the positive side.
        let (sel_pos, sel_neg) = downsample_one_uid(&pos, &[], 8);
        assert_eq!(sel_pos.len(), 8);
        assert!(sel_neg.is_empty(), "no negatives selected");

        // Too few overall: keep everything available.
        let (sel_pos, sel_neg) = downsample_one_uid(&pos[..2], &neg[..1], 8);
        assert_eq!(sel_pos.len(), 2);
        assert_eq!(sel_neg.len(), 1);

        // A target of zero selects nothing.
        let (sel_pos, sel_neg) = downsample_one_uid(&pos, &neg, 0);
        assert!(sel_pos.is_empty() && sel_neg.is_empty());
    }

    #[test]
    fn adaptive_sampling_covers_the_union_of_prompts() {
        let pos = cache(&[("a", &[1.0; 10]), ("b", &[1.0])]);
        let neg = cache(&[("a", &[0.0; 10]), ("c", &[0.0; 10])]);
        let (out_pos, out_neg) = sampling_ada(&pos, &neg, 8);
        assert_eq!(out_pos.len(), 3);
        assert_eq!(out_neg.len(), 3);
        assert_eq!(out_pos["a"].len(), 4);
        assert_eq!(out_neg["a"].len(), 4);
        assert_eq!(out_pos["b"].len(), 1);
        assert_eq!(out_neg["c"].len(), 8);
        assert!(out_pos["c"].is_empty(), "no positives for c");
        assert!(out_neg["b"].is_empty(), "no negatives for b");
    }

    #[test]
    fn the_registry_pre_registers_every_builtin() {
        let registry = ProcessorsRegistry::new();
        assert_eq!(registry.classifier_names(), vec![DEFAULT_CLASSIFIER]);
        assert_eq!(
            registry.validator_names(),
            vec![DEFAULT_VALIDATOR, BALANCED_VALIDATOR]
        );
        assert_eq!(
            registry.sampler_names(),
            vec![DEFAULT_SAMPLER, ADAPTIVE_SAMPLER]
        );

        let classifier = registry.classifier(DEFAULT_CLASSIFIER).expect("builtin");
        let (pos, neg) = classifier(&[rollout(0, 1.0), rollout(1, 0.0)]);
        assert_eq!((pos.len(), neg.len()), (1, 1));

        let validator = registry.validator(BALANCED_VALIDATOR).expect("builtin");
        assert!(validator(
            &[rollout(0, 1.0), rollout(1, 1.0)],
            &[rollout(2, 0.0)],
            Some(2)
        ));

        let sampler = registry.sampler(ADAPTIVE_SAMPLER).expect("builtin");
        let (pos, neg) = sampler(
            &cache(&[("a", &[1.0; 4])]),
            &cache(&[("a", &[0.0; 4])]),
            Some(2),
        );
        assert_eq!(pos["a"].len(), 1);
        assert_eq!(neg["a"].len(), 1);
    }

    #[test]
    fn unknown_names_report_the_registered_set() {
        let registry = ProcessorsRegistry::new();
        let error = registry.classifier("nope").err().expect("unknown");
        assert!(error.to_string().contains("unknown classifier \"nope\""));
        assert!(error.to_string().contains("default_classify_rollouts"));
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );

        let error = registry.validator("nope").err().expect("unknown");
        assert!(error.to_string().contains("validate_stop_balanced"));

        let error = registry.sampler("nope").err().expect("unknown");
        assert!(error.to_string().contains("sampling_ada"));
    }

    #[test]
    fn registration_shadows_a_builtin_and_returns_it() {
        let mut registry = ProcessorsRegistry::new();
        let previous = registry.register_classifier(DEFAULT_CLASSIFIER, |rollouts| {
            (rollouts.to_vec(), Vec::new())
        });
        assert!(previous.is_some());
        let classifier = registry.classifier(DEFAULT_CLASSIFIER).expect("shadowed");
        let (pos, neg) = classifier(&[rollout(0, 0.0)]);
        assert_eq!(pos.len(), 1);
        assert!(neg.is_empty(), "everything is positive");

        let previous_validator =
            registry.register_validator("custom", |_pos, _neg, keep| keep == Some(3));
        assert!(previous_validator.is_none());
        let validator = registry.validator("custom").expect("custom");
        assert!(validator(&[], &[], Some(3)));
        assert_eq!(
            registry.validator_names(),
            vec![
                "custom".to_owned(),
                DEFAULT_VALIDATOR.to_owned(),
                BALANCED_VALIDATOR.to_owned()
            ]
        );
        assert!(registry.sampler(DEFAULT_SAMPLER).is_ok());
        let debug = format!("{registry:?}");
        assert!(debug.contains("custom"));
    }
}
