//! [`CaseLoader`]: an in-memory collection of [`Case`]s with deterministic
//! shuffling and train/test splitting.
//!
//! Ports `CaseLoader` from the Python `openjiuwen/dev_tools/tune/`. The Python
//! loader uses `random.shuffle`; here the shuffle is driven by the module's one
//! `Rng64` `xorshift64*` generator (in `tune::prng`) so the same seed always
//! yields the same order and the crate needs no random-number dependency.
//!
//! Two cuts are offered. [`CaseLoader::split`] returns two plain loaders, which
//! is what the tuning optimizers take; [`CaseLoader::split_tagged`] performs the
//! identical cut and additionally tags the halves [`CaseSplit::Train`] and
//! [`CaseSplit::Val`], so a score measured on one of them keeps its provenance.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::tune::base::Case;
use crate::tune::dataset::{CaseSplit, SplitDataset};
use crate::tune::prng::Rng64;

/// A collection of tuning [`Case`]s.
///
/// Python: `openjiuwen.dev_tools.tune.CaseLoader`. Construction is infallible;
/// ratio arguments are validated by [`CaseLoader::split`] and
/// [`CaseLoader::split_tagged`].
///
/// Serializes to a single-key object holding the cases in their current order,
/// for example `{"cases": [{...}, {...}]}`. Deserialization is permissive: the
/// `cases` key may be omitted (yielding an empty loader) and unknown keys are
/// ignored.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CaseLoader {
    /// The contained cases, in their current order.
    cases: Vec<Case>,
}

impl CaseLoader {
    /// Collects cases into a loader, preserving their order.
    pub fn new(cases: impl IntoIterator<Item = Case>) -> Self {
        Self {
            cases: cases.into_iter().collect(),
        }
    }

    /// The number of cases.
    #[must_use]
    pub fn size(&self) -> usize {
        self.cases.len()
    }

    /// Whether the loader holds no cases.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }

    /// The cases in their current order.
    #[must_use]
    pub fn cases(&self) -> &[Case] {
        &self.cases
    }

    /// Consumes the loader, returning the contained cases.
    #[must_use]
    pub fn into_cases(self) -> Vec<Case> {
        self.cases
    }

    /// The case at `index`, or `None` when the index is out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Case> {
        self.cases.get(index)
    }

    /// Iterates the cases by reference, in their current order.
    pub fn iter(&self) -> std::slice::Iter<'_, Case> {
        self.cases.iter()
    }

    /// Returns a copy of the loader with the cases in a shuffled order.
    ///
    /// A deterministic Fisher-Yates shuffle driven by the module's shared
    /// `Rng64` generator: the same `random_seed` always produces the same
    /// permutation of the same cases.
    #[must_use]
    pub fn shuffle(&self, random_seed: u64) -> Self {
        let mut cases = self.cases.clone();
        Rng64::new(random_seed).shuffle(&mut cases);
        Self { cases }
    }

    /// Splits the loader into a left and a right part at `floor(ratio * size)`.
    ///
    /// `ratio` `0.0` puts every case on the right, `1.0` puts every case on the
    /// left. The left part is `[0, split)`, the right part `[split, size)`.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `ratio` is not finite or outside `0.0..=1.0`.
    pub fn split(&self, ratio: f64) -> Result<(Self, Self)> {
        if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
            return Err(Error::validation(format!(
                "CaseLoader.split ratio must be finite and between 0.0 and 1.0, got {ratio}"
            )));
        }
        let split_at = (ratio * self.cases.len() as f64).floor() as usize;
        let (left, right) = self.cases.split_at(split_at);
        Ok((Self::new(left.to_vec()), Self::new(right.to_vec())))
    }

    /// Splits the loader exactly like [`split`](Self::split) and tags the halves.
    ///
    /// The tagged sibling of [`split`](Self::split), which stays untagged and
    /// unchanged: the cut is still `floor(ratio * size)`, but the left part
    /// `[0, split)` comes back as a [`SplitDataset`] tagged [`CaseSplit::Train`]
    /// and the right part `[split, size)` as one tagged [`CaseSplit::Val`], so
    /// the provenance travels with the cases into
    /// [`Trainer::train_split`](crate::tune::Trainer::train_split) and into the
    /// [`EvaluationSummary`](crate::tune::EvaluationSummary) it reports.
    ///
    /// `ratio` `0.0` puts every case on the val side (an empty train split) and
    /// `1.0` puts every case on the train side (an empty val split).
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `ratio` is not finite or outside `0.0..=1.0`,
    /// with the same message as [`split`](Self::split).
    pub fn split_tagged(&self, ratio: f64) -> Result<(SplitDataset, SplitDataset)> {
        let (train, val) = self.split(ratio)?;
        Ok((
            SplitDataset::new(CaseSplit::Train, train.into_cases()),
            SplitDataset::new(CaseSplit::Val, val.into_cases()),
        ))
    }
}

impl IntoIterator for CaseLoader {
    type Item = Case;
    type IntoIter = std::vec::IntoIter<Case>;

    /// Iterates the cases by value, in their current order.
    fn into_iter(self) -> Self::IntoIter {
        self.cases.into_iter()
    }
}

impl<'a> IntoIterator for &'a CaseLoader {
    type Item = &'a Case;
    type IntoIter = std::slice::Iter<'a, Case>;

    /// Iterates the cases by reference, in their current order.
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::Value;

    use super::*;

    /// A minimal valid case with the given identifier.
    fn case(id: &str) -> Case {
        let mut inputs = BTreeMap::new();
        inputs.insert("question".to_owned(), Value::String(id.to_owned()));
        let mut label = BTreeMap::new();
        label.insert("answer".to_owned(), Value::String(id.to_owned()));
        Case::new(inputs, label).unwrap().with_case_id(id)
    }

    /// A loader holding `count` cases with the ids `0..count`.
    fn loader(count: usize) -> CaseLoader {
        CaseLoader::new((0..count).map(|i| case(&i.to_string())))
    }

    /// The case ids of a loader, in order.
    fn ids(loader: &CaseLoader) -> Vec<String> {
        loader.iter().map(|c| c.case_id.clone()).collect()
    }

    #[test]
    fn new_tracks_size_and_emptiness() {
        let empty = CaseLoader::new(Vec::new());
        assert_eq!(empty.size(), 0);
        assert!(empty.is_empty());
        assert_eq!(empty.cases(), []);

        let populated = loader(3);
        assert_eq!(populated.size(), 3);
        assert!(!populated.is_empty());
        assert_eq!(ids(&populated), ["0", "1", "2"]);
    }

    #[test]
    fn into_cases_returns_the_cases_in_order() {
        let cases = loader(2).into_cases();
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].case_id, "0");
        assert_eq!(cases[1].case_id, "1");
    }

    #[test]
    fn get_reports_in_and_out_of_range_indexes() {
        let loader = loader(2);
        assert_eq!(loader.get(0).map(|c| c.case_id.as_str()), Some("0"));
        assert_eq!(loader.get(1).map(|c| c.case_id.as_str()), Some("1"));
        assert!(loader.get(2).is_none());
        assert!(loader.get(usize::MAX).is_none());
    }

    #[test]
    fn split_at_zero_ratio_yields_an_empty_left_part() {
        let (left, right) = loader(3).split(0.0).unwrap();
        assert_eq!(left.size(), 0);
        assert_eq!(ids(&right), ["0", "1", "2"]);
    }

    #[test]
    fn split_at_full_ratio_yields_an_empty_right_part() {
        let (left, right) = loader(3).split(1.0).unwrap();
        assert_eq!(ids(&left), ["0", "1", "2"]);
        assert_eq!(right.size(), 0);
    }

    #[test]
    fn split_uses_the_floor_of_ratio_times_size() {
        let (left, right) = loader(5).split(0.5).unwrap();
        assert_eq!(ids(&left), ["0", "1"]);
        assert_eq!(ids(&right), ["2", "3", "4"]);

        let (left, right) = loader(4).split(0.5).unwrap();
        assert_eq!(left.size(), 2);
        assert_eq!(right.size(), 2);

        let (left, right) = loader(3).split(1.0 / 3.0).unwrap();
        assert_eq!(left.size(), 1);
        assert_eq!(right.size(), 2);
    }

    #[test]
    fn split_rejects_invalid_ratios() {
        for ratio in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = loader(3).split(ratio).unwrap_err();
            assert!(
                matches!(&error, Error::Validation(message) if message.contains("ratio")),
                "unexpected error for {ratio}: {error}"
            );
        }
    }

    #[test]
    fn split_of_an_empty_loader_yields_two_empty_loaders() {
        let (left, right) = CaseLoader::default().split(0.5).unwrap();
        assert!(left.is_empty());
        assert!(right.is_empty());
    }

    #[test]
    fn split_tagged_tags_the_halves_train_and_val() {
        let (train, val) = loader(4).split_tagged(0.5).unwrap();
        assert_eq!(train.split, CaseSplit::Train);
        assert_eq!(val.split, CaseSplit::Val);
        assert_eq!(train.len(), 2);
        assert_eq!(val.len(), 2);
    }

    #[test]
    fn split_tagged_matches_the_untagged_split() {
        for ratio in [0.0, 0.25, 0.5, 1.0 / 3.0, 0.75, 1.0] {
            for count in [0, 1, 3, 5, 7] {
                let loader = loader(count);
                let (left, right) = loader.split(ratio).unwrap();
                let (train, val) = loader.split_tagged(ratio).unwrap();
                assert_eq!(train.len(), left.size(), "ratio {ratio}, count {count}");
                assert_eq!(val.len(), right.size(), "ratio {ratio}, count {count}");
                assert_eq!(
                    train.cases(),
                    left.cases(),
                    "ratio {ratio}, count {count} keeps the train cases"
                );
                assert_eq!(
                    val.cases(),
                    right.cases(),
                    "ratio {ratio}, count {count} keeps the val cases"
                );
            }
        }
    }

    #[test]
    fn split_tagged_at_zero_ratio_puts_everything_on_the_val_side() {
        let (train, val) = loader(3).split_tagged(0.0).unwrap();
        assert!(train.is_empty());
        assert_eq!(train.split, CaseSplit::Train);
        assert_eq!(
            val.iter().map(|c| c.case_id.as_str()).collect::<Vec<_>>(),
            ["0", "1", "2"]
        );
        assert_eq!(val.split, CaseSplit::Val);
    }

    #[test]
    fn split_tagged_at_full_ratio_puts_everything_on_the_train_side() {
        let (train, val) = loader(3).split_tagged(1.0).unwrap();
        assert_eq!(
            train.iter().map(|c| c.case_id.as_str()).collect::<Vec<_>>(),
            ["0", "1", "2"]
        );
        assert_eq!(train.split, CaseSplit::Train);
        assert!(val.is_empty());
        assert_eq!(val.split, CaseSplit::Val);
    }

    #[test]
    fn split_tagged_of_an_empty_loader_yields_two_empty_tagged_datasets() {
        let (train, val) = CaseLoader::default().split_tagged(0.5).unwrap();
        assert!(train.is_empty());
        assert_eq!(train.split, CaseSplit::Train);
        assert!(val.is_empty());
        assert_eq!(val.split, CaseSplit::Val);
    }

    #[test]
    fn split_tagged_rejects_invalid_ratios() {
        for ratio in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = loader(3).split_tagged(ratio).unwrap_err();
            assert!(
                matches!(&error, Error::Validation(message) if message.contains("ratio")),
                "unexpected error for {ratio}: {error}"
            );
        }
    }

    #[test]
    fn shuffle_is_deterministic_for_a_seed() {
        let loader = loader(6);
        let first = loader.shuffle(42);
        let second = loader.shuffle(42);
        assert_eq!(ids(&first), ids(&second));
        assert_eq!(first, second);
    }

    #[test]
    fn shuffle_is_a_permutation_of_the_input() {
        let loader = loader(7);
        for seed in [0, 1, 2, 42, u64::MAX] {
            let shuffled = loader.shuffle(seed);
            assert_eq!(shuffled.size(), loader.size());
            let mut ids = ids(&shuffled);
            ids.sort();
            let mut expected: Vec<String> = (0..7).map(|i| i.to_string()).collect();
            expected.sort();
            assert_eq!(ids, expected, "seed {seed} keeps the multiset");
        }
    }

    #[test]
    fn shuffle_actually_reorders_for_some_seed() {
        let loader = loader(8);
        let original = ids(&loader);
        let varied = (0..8u64)
            .map(|seed| ids(&loader.shuffle(seed)))
            .any(|candidate| candidate != original);
        assert!(varied, "at least one seed must change the order");
    }

    #[test]
    fn shuffle_handles_tiny_loaders() {
        let empty = CaseLoader::default().shuffle(7);
        assert!(empty.is_empty());

        let single = loader(1).shuffle(7);
        assert_eq!(ids(&single), ["0"]);
    }

    #[test]
    fn iteration_by_value_and_by_reference_preserve_order() {
        let loader = loader(3);
        let by_reference: Vec<&str> = (&loader).into_iter().map(|c| c.case_id.as_str()).collect();
        assert_eq!(by_reference, ["0", "1", "2"]);

        let by_value: Vec<String> = loader.into_iter().map(|c| c.case_id).collect();
        assert_eq!(by_value, ["0", "1", "2"]);
    }

    #[test]
    fn serialization_round_trips_through_the_cases_key() {
        let loader = loader(3);

        let value = serde_json::to_value(&loader).unwrap();
        let cases = value
            .get("cases")
            .and_then(Value::as_array)
            .expect("the JSON object must hold a `cases` array");
        assert_eq!(cases.len(), 3);
        assert_eq!(value.as_object().map(serde_json::Map::len), Some(1));

        let restored: CaseLoader = serde_json::from_value(value).unwrap();
        assert_eq!(restored, loader);
        assert_eq!(ids(&restored), ["0", "1", "2"]);
        assert_eq!(ids(&restored), ids(&loader));
    }

    #[test]
    fn deserialization_defaults_an_empty_loader_and_ignores_unknown_keys() {
        let empty: CaseLoader = serde_json::from_str("{}").unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty, CaseLoader::default());

        let with_extra: CaseLoader = serde_json::from_value(serde_json::json!({
            "cases": [serde_json::to_value(case("0")).unwrap()],
            "unknown_key": {"nested": true},
        }))
        .unwrap();
        assert_eq!(ids(&with_extra), ["0"]);
    }
}
