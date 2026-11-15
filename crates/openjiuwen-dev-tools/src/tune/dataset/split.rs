//! [`CaseSplit`] and [`SplitDataset`]: a case list tagged with the partition it
//! came from.
//!
//! The Python tune package hands a bare `list[Case]` to a trainer and loses the
//! information about which half of a split it is looking at. Here the tag travels
//! with the cases, so a report or an evaluation summary can say whether a score
//! was measured on the training or the validation partition.
//!
//! [`SplitDataset::split`] mirrors [`CaseLoader::split`](crate::tune::CaseLoader::split)
//! exactly — the same `floor(ratio * size)` cut and the same ratio validation —
//! and keeps the parent tag on both halves, because a partition of a training set
//! is still training data.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::tune::base::Case;
use crate::tune::dataset::batch::{BatchScheduler, Epoch};
use crate::tune::dataset::sampler::CaseSampler;

/// The partition a case set (or a score measured over one) came from.
///
/// Serializes as `"train"`, `"val"`, or `"unspecified"` (snake case, matching the
/// Python spelling). [`CaseSplit::Unspecified`] is the default and is what an
/// untagged evaluation reports: a case list that never went through a split still
/// has provenance, namely "no partition was assigned".
///
/// The [`Ord`] order follows the declaration order (`Train` < `Val` <
/// `Unspecified`), so the type can key an ordered map without a custom comparator.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum CaseSplit {
    /// The training partition: cases an optimizer or trainer may fit against.
    Train,
    /// The validation partition: cases used to score, never to fit.
    Val,
    /// No partition was assigned, which is what an untagged evaluation reports.
    #[default]
    Unspecified,
}

impl CaseSplit {
    /// The Python spelling of the tag: `"train"`, `"val"`, or `"unspecified"`.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Val => "val",
            Self::Unspecified => "unspecified",
        }
    }
}

impl std::fmt::Display for CaseSplit {
    /// Writes [`CaseSplit::as_str`], so `format!("{split}")` matches the JSON tag.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A [`Case`] list bundled with the [`CaseSplit`] it belongs to.
///
/// Serializes to a two-key object, `{"split": "train", "cases": [{...}, {...}]}`.
/// Deserialization is permissive: either key may be omitted — a missing `cases`
/// yields an empty dataset and a missing `split` yields
/// [`CaseSplit::Unspecified`] — and unknown keys are ignored, so a legacy payload
/// that only held a case array still loads.
///
/// The accessors mirror [`CaseLoader`](crate::tune::CaseLoader)'s shapes so the two
/// collections can be used interchangeably where only the cases matter.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SplitDataset {
    /// The partition every case in [`SplitDataset::cases`] was taken from.
    pub split: CaseSplit,
    /// The contained cases, in their current order.
    pub cases: Vec<Case>,
}

impl SplitDataset {
    /// Collects cases into a dataset tagged with `split`, preserving their order.
    #[must_use]
    pub fn new(split: CaseSplit, cases: impl IntoIterator<Item = Case>) -> Self {
        Self {
            split,
            cases: cases.into_iter().collect(),
        }
    }

    /// The number of cases.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cases.len()
    }

    /// Whether the dataset holds no cases.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }

    /// The cases in their current order.
    #[must_use]
    pub fn cases(&self) -> &[Case] {
        &self.cases
    }

    /// Consumes the dataset, returning the contained cases.
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

    /// Returns a copy of the dataset retagged with `split`, keeping the cases.
    ///
    /// Used to mark the result of a re-partitioning step (for example, the
    /// validation half handed to a different trainer) without rebuilding the list.
    #[must_use]
    pub fn tag(&self, split: CaseSplit) -> Self {
        Self {
            split,
            cases: self.cases.clone(),
        }
    }

    /// Splits the dataset into a left and a right part at `floor(ratio * size)`.
    ///
    /// `ratio` `0.0` puts every case on the right, `1.0` puts every case on the
    /// left. The left part is `[0, split)`, the right part `[split, size)`.
    /// Both halves keep this dataset's [`split`](SplitDataset::split) tag: the
    /// partition a case came from does not change when the list is cut again.
    /// Which half is the training set is the caller's decision, so the tag is
    /// deliberately not rewritten here — use [`SplitDataset::tag`] for that.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `ratio` is not finite or outside `0.0..=1.0`.
    pub fn split(&self, ratio: f64) -> Result<(Self, Self)> {
        if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
            return Err(Error::validation(format!(
                "SplitDataset.split ratio must be finite and between 0.0 and 1.0, got {ratio}"
            )));
        }
        let split_at = (ratio * self.cases.len() as f64).floor() as usize;
        let (left, right) = self.cases.split_at(split_at);
        Ok((
            Self::new(self.split, left.iter().cloned()),
            Self::new(self.split, right.iter().cloned()),
        ))
    }

    /// Samples the dataset's cases through `sampler`.
    ///
    /// A thin convenience over [`CaseSampler::sample`], so a tagged partition can
    /// be subsampled without reaching for [`SplitDataset::cases`] first. The
    /// sampler's clamping, zero-count, determinism, and identity contract all
    /// apply unchanged; the tag is not carried into the returned cases.
    ///
    /// # Errors
    ///
    /// Whatever the sampler reports; the samplers in this module never fail.
    pub fn sample(&self, sampler: &dyn CaseSampler) -> Result<Vec<Case>> {
        sampler.sample(self.cases())
    }

    /// Plans the dataset's case list into epochs and batches with `scheduler`.
    ///
    /// A thin convenience over [`BatchScheduler::plan`], so a tagged partition can
    /// be turned straight into a trainer's schedule. The plan is a pure function
    /// of the cases and the scheduler's configuration.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the scheduler's `batch_size` or `epochs` is `0`.
    pub fn batches(&self, scheduler: &BatchScheduler) -> Result<Vec<Epoch>> {
        scheduler.plan(self.cases())
    }
}

impl IntoIterator for SplitDataset {
    type Item = Case;
    type IntoIter = std::vec::IntoIter<Case>;

    /// Iterates the cases by value, in their current order.
    fn into_iter(self) -> Self::IntoIter {
        self.cases.into_iter()
    }
}

impl<'a> IntoIterator for &'a SplitDataset {
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
    use crate::tune::dataset::{RandomSampler, SequentialSampler};

    /// A minimal valid case with the given identifier.
    fn case(id: &str) -> Case {
        let mut inputs = BTreeMap::new();
        inputs.insert("question".to_owned(), Value::String(id.to_owned()));
        let mut label = BTreeMap::new();
        label.insert("answer".to_owned(), Value::String(id.to_owned()));
        Case::new(inputs, label).unwrap().with_case_id(id)
    }

    /// A dataset of `count` cases with the ids `0..count`, tagged `split`.
    fn dataset(split: CaseSplit, count: usize) -> SplitDataset {
        SplitDataset::new(split, (0..count).map(|i| case(&i.to_string())))
    }

    /// The case ids of a dataset, in order.
    fn ids(dataset: &SplitDataset) -> Vec<String> {
        dataset.iter().map(|c| c.case_id.clone()).collect()
    }

    #[test]
    fn new_tracks_split_length_and_emptiness() {
        let empty = SplitDataset::new(CaseSplit::Val, Vec::new());
        assert_eq!(empty.split, CaseSplit::Val);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
        assert_eq!(empty.cases(), []);

        let populated = dataset(CaseSplit::Train, 3);
        assert_eq!(populated.split, CaseSplit::Train);
        assert_eq!(populated.len(), 3);
        assert!(!populated.is_empty());
        assert_eq!(ids(&populated), ["0", "1", "2"]);
    }

    #[test]
    fn default_is_the_unspecified_tag_with_no_cases() {
        let dataset = SplitDataset::default();
        assert_eq!(dataset.split, CaseSplit::Unspecified);
        assert!(dataset.is_empty());
        assert_eq!(CaseSplit::default(), CaseSplit::Unspecified);
    }

    #[test]
    fn into_cases_returns_the_cases_in_order() {
        let cases = dataset(CaseSplit::Train, 2).into_cases();
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].case_id, "0");
        assert_eq!(cases[1].case_id, "1");
    }

    #[test]
    fn get_reports_in_and_out_of_range_indexes() {
        let dataset = dataset(CaseSplit::Val, 2);
        assert_eq!(dataset.get(0).map(|c| c.case_id.as_str()), Some("0"));
        assert_eq!(dataset.get(1).map(|c| c.case_id.as_str()), Some("1"));
        assert!(dataset.get(2).is_none());
        assert!(dataset.get(usize::MAX).is_none());
    }

    #[test]
    fn tag_retags_a_copy_and_keeps_the_cases() {
        let dataset = dataset(CaseSplit::Train, 2);
        let retagged = dataset.tag(CaseSplit::Val);
        assert_eq!(retagged.split, CaseSplit::Val);
        assert_eq!(ids(&retagged), ids(&dataset));
        assert_eq!(retagged.cases, dataset.cases);
        // The original is untouched.
        assert_eq!(dataset.split, CaseSplit::Train);
    }

    #[test]
    fn iteration_by_value_and_by_reference_preserve_order() {
        let dataset = dataset(CaseSplit::Train, 3);
        let by_reference: Vec<&str> = (&dataset).into_iter().map(|c| c.case_id.as_str()).collect();
        assert_eq!(by_reference, ["0", "1", "2"]);

        let by_value: Vec<String> = dataset.into_iter().map(|c| c.case_id).collect();
        assert_eq!(by_value, ["0", "1", "2"]);
    }

    #[test]
    fn split_at_zero_ratio_yields_an_empty_left_part() {
        let (left, right) = dataset(CaseSplit::Train, 3).split(0.0).unwrap();
        assert!(left.is_empty());
        assert_eq!(ids(&right), ["0", "1", "2"]);
    }

    #[test]
    fn split_at_full_ratio_yields_an_empty_right_part() {
        let (left, right) = dataset(CaseSplit::Val, 3).split(1.0).unwrap();
        assert_eq!(ids(&left), ["0", "1", "2"]);
        assert!(right.is_empty());
    }

    #[test]
    fn split_uses_the_floor_of_ratio_times_size() {
        let (left, right) = dataset(CaseSplit::Train, 5).split(0.5).unwrap();
        assert_eq!(ids(&left), ["0", "1"]);
        assert_eq!(ids(&right), ["2", "3", "4"]);

        let (left, right) = dataset(CaseSplit::Train, 4).split(0.5).unwrap();
        assert_eq!(left.len(), 2);
        assert_eq!(right.len(), 2);

        let (left, right) = dataset(CaseSplit::Train, 3).split(1.0 / 3.0).unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(right.len(), 2);
    }

    #[test]
    fn both_split_halves_keep_the_parent_tag() {
        for split in [CaseSplit::Train, CaseSplit::Val, CaseSplit::Unspecified] {
            let (left, right) = dataset(split, 4).split(0.5).unwrap();
            assert_eq!(left.split, split);
            assert_eq!(right.split, split);
        }
    }

    #[test]
    fn split_rejects_invalid_ratios() {
        for ratio in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = dataset(CaseSplit::Train, 3).split(ratio).unwrap_err();
            assert!(
                matches!(&error, Error::Validation(message) if message.contains("ratio")),
                "unexpected error for {ratio}: {error}"
            );
        }
    }

    #[test]
    fn split_of_an_empty_dataset_yields_two_empty_datasets() {
        let (left, right) = SplitDataset::default().split(0.5).unwrap();
        assert!(left.is_empty());
        assert!(right.is_empty());
        assert_eq!(left.split, CaseSplit::Unspecified);
        assert_eq!(right.split, CaseSplit::Unspecified);
    }

    #[test]
    fn sample_and_batches_delegate_to_the_seams() {
        let dataset = dataset(CaseSplit::Train, 5);

        // `sample` hands the dataset's cases to the sampler unchanged.
        let sampled = dataset.sample(&SequentialSampler::new(2)).unwrap();
        assert_eq!(sampled, dataset.cases()[..2].to_vec());
        let all = dataset.sample(&RandomSampler::new(7)).unwrap();
        assert_eq!(all.len(), dataset.len());

        // `batches` plans exactly what the scheduler plans over the same cases.
        let scheduler = BatchScheduler::new(2).unwrap();
        assert_eq!(
            dataset.batches(&scheduler).unwrap(),
            scheduler.plan(dataset.cases()).unwrap()
        );
        let plan = dataset.batches(&scheduler).unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].batches.len(), 3);
        assert_eq!(plan[0].batches[2].indices, [4]);
        let last = &plan[0].batches[2];
        assert_eq!(
            last.cases
                .iter()
                .map(|case| case.case_id.as_str())
                .collect::<Vec<_>>(),
            ["4"]
        );

        // A scheduler that cannot plan reports its validation error unchanged.
        let broken = BatchScheduler::new(2).unwrap().with_epochs(0);
        assert!(dataset.batches(&broken).is_err());
    }

    #[test]
    fn case_split_serializes_to_the_python_spelling() {
        for (split, expected) in [
            (CaseSplit::Train, "\"train\""),
            (CaseSplit::Val, "\"val\""),
            (CaseSplit::Unspecified, "\"unspecified\""),
        ] {
            assert_eq!(serde_json::to_string(&split).unwrap(), expected);
            let restored: CaseSplit = serde_json::from_str(expected).unwrap();
            assert_eq!(restored, split);
            assert_eq!(format!("\"{}\"", split.as_str()), expected);
        }
    }

    #[test]
    fn display_matches_as_str() {
        for split in [CaseSplit::Train, CaseSplit::Val, CaseSplit::Unspecified] {
            assert_eq!(split.to_string(), split.as_str());
        }
        assert_eq!(CaseSplit::Train.to_string(), "train");
        assert_eq!(CaseSplit::Val.to_string(), "val");
        assert_eq!(CaseSplit::Unspecified.to_string(), "unspecified");
    }

    #[test]
    fn ordering_is_usable_as_a_btree_map_key() {
        let mut map = BTreeMap::new();
        map.insert(CaseSplit::Unspecified, "none");
        map.insert(CaseSplit::Val, "validation");
        map.insert(CaseSplit::Train, "training");
        map.insert(CaseSplit::Train, "training again");

        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&CaseSplit::Train), Some(&"training again"));
        let keys: Vec<CaseSplit> = map.keys().copied().collect();
        assert_eq!(
            keys,
            [CaseSplit::Train, CaseSplit::Val, CaseSplit::Unspecified]
        );
        assert!(CaseSplit::Train < CaseSplit::Val);
        assert!(CaseSplit::Val < CaseSplit::Unspecified);
        assert_eq!(
            CaseSplit::Train.cmp(&CaseSplit::Train),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn dataset_round_trips_through_the_split_and_cases_keys() {
        let dataset = dataset(CaseSplit::Val, 3);

        let value = serde_json::to_value(&dataset).unwrap();
        assert_eq!(value.get("split").and_then(Value::as_str), Some("val"));
        let cases = value
            .get("cases")
            .and_then(Value::as_array)
            .expect("the JSON object must hold a `cases` array");
        assert_eq!(cases.len(), 3);
        assert_eq!(value.as_object().map(serde_json::Map::len), Some(2));

        let restored: SplitDataset = serde_json::from_value(value).unwrap();
        assert_eq!(restored, dataset);
        assert_eq!(ids(&restored), ["0", "1", "2"]);
    }

    #[test]
    fn deserialization_defaults_missing_keys_and_ignores_unknown_ones() {
        let empty: SplitDataset = serde_json::from_str("{}").unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.split, CaseSplit::Unspecified);
        assert_eq!(empty, SplitDataset::default());

        // A payload that lost its `split` key is still a validation set's cases.
        let partial: SplitDataset = serde_json::from_value(serde_json::json!({
            "cases": [serde_json::to_value(case("0")).unwrap()],
        }))
        .unwrap();
        assert_eq!(partial.split, CaseSplit::Unspecified);
        assert_eq!(ids(&partial), ["0"]);

        // A payload that lost its `cases` key is an empty, still-tagged dataset.
        let tagged: SplitDataset = serde_json::from_str("{\"split\": \"train\"}").unwrap();
        assert_eq!(tagged.split, CaseSplit::Train);
        assert!(tagged.is_empty());

        let with_extra: SplitDataset = serde_json::from_value(serde_json::json!({
            "split": "val",
            "cases": [serde_json::to_value(case("0")).unwrap()],
            "unknown_key": {"nested": true},
        }))
        .unwrap();
        assert_eq!(with_extra.split, CaseSplit::Val);
        assert_eq!(ids(&with_extra), ["0"]);
    }
}
