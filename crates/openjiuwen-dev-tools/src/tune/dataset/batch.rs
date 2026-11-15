//! [`BatchScheduler`]: the epoch/batch plan a trainer iterates.
//!
//! Ports the batch-scheduling layer the Python tune trainer keeps above
//! `dataset/case_loader.py`: a batch size, an epoch count, an optional shuffle
//! seed, and a `drop_last` flag produce a deterministic
//! `Vec<`[`Epoch`]`>` of [`Batch`]es. The plan is data, not an iterator: every
//! call to [`BatchScheduler::plan`] rebuilds the whole schedule from the case
//! list, so the trainer can inspect it (or replay it) before running a step.
//!
//! Shuffling is per epoch and derived from `(seed, epoch)`, so a shuffled plan is
//! reproducible yet different epochs see different orders, and every epoch is a
//! permutation of the input — a batch never repeats or invents a case.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::tune::base::Case;
use crate::tune::prng::Rng64;

/// The epoch count a fresh [`BatchScheduler`] plans.
///
/// Python's trainer defaults to one pass over the data, and so does this layer;
/// multi-epoch training opts in with [`BatchScheduler::with_epochs`].
pub const DEFAULT_EPOCHS: usize = 1;

/// The stride that derives a per-epoch shuffle seed from the scheduler seed.
///
/// Epoch `e` is shuffled with a generator seeded from
/// `seed.wrapping_add((e as u64).wrapping_mul(EPOCH_SEED_STRIDE))`, so the same
/// seed always gives the same per-epoch order and a new epoch does not replay the
/// previous one.
const EPOCH_SEED_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

/// How many epochs to run, how large a batch is, and whether to shuffle.
///
/// Serializes as the four-key object
/// `{"batch_size": 4, "epochs": 2, "shuffle_seed": null, "drop_last": false}`;
/// deserialization is permissive (any missing key falls back to
/// [`BatchScheduler::default`]) and unknown keys are ignored. Deserializing can
/// produce a scheduler whose `batch_size` or `epochs` is `0`;
/// [`BatchScheduler::plan`] rejects that instead of dividing by zero.
///
/// [`BatchScheduler::default`] is the `batch_size = 1` configuration: one epoch,
/// no shuffle (the input order), and the last partial batch kept.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct BatchScheduler {
    /// The maximum number of cases in one batch; must be at least `1` to plan.
    batch_size: usize,
    /// The number of passes over the case list; must be at least `1` to plan.
    epochs: usize,
    /// The shuffle seed, or `None` to keep the input order in every epoch.
    shuffle_seed: Option<u64>,
    /// Whether to drop the final partial batch of each epoch.
    drop_last: bool,
}

impl BatchScheduler {
    /// Creates a scheduler with `batch_size`, one epoch, no shuffle, and the last
    /// partial batch kept.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `batch_size` is `0`, naming `batch_size`; a
    /// zero-sized batch cannot be planned.
    pub fn new(batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(Error::validation(
                "BatchScheduler batch_size must be at least 1, got 0",
            ));
        }
        Ok(Self {
            batch_size,
            epochs: DEFAULT_EPOCHS,
            shuffle_seed: None,
            drop_last: false,
        })
    }

    /// Returns a copy of the scheduler planning `epochs` passes.
    ///
    /// `0` is not rejected here — the builder stays infallible — but
    /// [`BatchScheduler::plan`] then reports the invalid epoch count.
    #[must_use]
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Returns a copy of the scheduler shuffling every epoch with `seed`.
    ///
    /// Each epoch derives its own generator from `(seed, epoch)`, so a seeded plan
    /// is reproducible and different epochs see different orders. Seed `0` is
    /// accepted and remapped by the crate's generator.
    #[must_use]
    pub fn with_shuffle(mut self, seed: u64) -> Self {
        self.shuffle_seed = Some(seed);
        self
    }

    /// Returns a copy of the scheduler dropping the final partial batch.
    #[must_use]
    pub fn with_drop_last(mut self) -> Self {
        self.drop_last = true;
        self
    }

    /// Returns a copy of the scheduler keeping the final partial batch.
    ///
    /// The inverse of [`BatchScheduler::with_drop_last`], so a configuration read
    /// from JSON or built by a chain can be reset without rebuilding.
    #[must_use]
    pub fn without_drop_last(mut self) -> Self {
        self.drop_last = false;
        self
    }

    /// The configured batch size.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// The configured epoch count.
    #[must_use]
    pub fn epochs(&self) -> usize {
        self.epochs
    }

    /// The configured shuffle seed, or `None` for the input order.
    #[must_use]
    pub fn shuffle_seed(&self) -> Option<u64> {
        self.shuffle_seed
    }

    /// Whether the final partial batch is dropped.
    #[must_use]
    pub fn drop_last(&self) -> bool {
        self.drop_last
    }

    /// How many batches one epoch of `len` cases produces.
    ///
    /// `len / batch_size` when [`drop_last`](BatchScheduler::drop_last) is set,
    /// otherwise `ceil(len / batch_size)` so the trailing partial batch is kept. A
    /// zero `batch_size` (only reachable by deserializing an invalid
    /// configuration) reports `0` rather than dividing by zero, and an empty case
    /// list always reports `0`.
    #[must_use]
    pub fn batches_per_epoch(&self, len: usize) -> usize {
        if self.batch_size == 0 {
            return 0;
        }
        if self.drop_last {
            len / self.batch_size
        } else {
            len.div_ceil(self.batch_size)
        }
    }

    /// Plans every epoch and batch over `cases`.
    ///
    /// Each epoch numbers its indices `0..cases.len()`, shuffles them with the
    /// epoch's derived seed when [`shuffle_seed`](BatchScheduler::shuffle_seed) is
    /// set, and slices the result into chunks of `batch_size`; the final partial
    /// chunk is kept unless [`drop_last`](BatchScheduler::drop_last) is set.
    /// Batches carry the cloned cases plus the input positions they came from, and
    /// `step` counts batches globally from `1` across the whole plan.
    ///
    /// The plan is a pure function of `(cases, configuration)`: two calls return
    /// equal `Batch`es, and an unshuffled plan is always the input order.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `batch_size` is `0` (defensive: only
    /// deserialization can produce it) or when `epochs` is `0`.
    pub fn plan(&self, cases: &[Case]) -> Result<Vec<Epoch>> {
        if self.batch_size == 0 {
            return Err(Error::validation(
                "BatchScheduler cannot plan with batch_size 0; use new(1) or larger",
            ));
        }
        if self.epochs == 0 {
            return Err(Error::validation(
                "BatchScheduler cannot plan with epochs 0; use with_epochs(1) or larger",
            ));
        }

        let mut epochs = Vec::with_capacity(self.epochs);
        let mut step = 0usize;
        for epoch in 0..self.epochs {
            let mut indices: Vec<usize> = (0..cases.len()).collect();
            if let Some(seed) = self.shuffle_seed {
                let epoch_seed = seed.wrapping_add((epoch as u64).wrapping_mul(EPOCH_SEED_STRIDE));
                Rng64::new(epoch_seed).shuffle(&mut indices);
            }

            let mut batches = Vec::with_capacity(self.batches_per_epoch(cases.len()));
            for (index, chunk) in indices.chunks(self.batch_size).enumerate() {
                if self.drop_last && chunk.len() < self.batch_size {
                    break;
                }
                step += 1;
                batches.push(Batch {
                    epoch,
                    index,
                    step,
                    cases: chunk.iter().map(|&at| cases[at].clone()).collect(),
                    indices: chunk.to_vec(),
                });
            }
            epochs.push(Epoch {
                index: epoch,
                batches,
            });
        }
        Ok(epochs)
    }
}

impl Default for BatchScheduler {
    /// The `new(1)` configuration: batch size `1`, [`DEFAULT_EPOCHS`] epoch, no
    /// shuffle, and no dropped batch.
    ///
    /// Batch size `1` is the neutral default because it needs no case count: a
    /// trainer that never configured batching still gets one case per step.
    fn default() -> Self {
        Self {
            batch_size: 1,
            epochs: DEFAULT_EPOCHS,
            shuffle_seed: None,
            drop_last: false,
        }
    }
}

/// One pass over the case list.
///
/// Serializes as `{"index": 0, "batches": [...]}`, with the epoch `index` 0-based
/// and its batches in plan order. [`Epoch::default`] is the empty epoch (index
/// `0`, no batches); the derive exists so the permissive `serde(default)`
/// container attribute can fill in a key a payload omitted.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Epoch {
    /// The 0-based position of this epoch in the plan.
    pub index: usize,
    /// The batches of this epoch, in plan order.
    pub batches: Vec<Batch>,
}

impl Epoch {
    /// The number of batches in this epoch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.batches.len()
    }

    /// Whether this epoch has no batches, which happens for an empty case list or
    /// a `drop_last` epoch shorter than one batch.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }
}

/// One batch of an epoch.
///
/// Serializes as `{"epoch": 0, "index": 0, "step": 1, "cases": [...],
/// "indices": [0, 3, 1]}`. `epoch` and `index` are 0-based, `index` counts within
/// its epoch, `step` is the global 1-based step across the whole plan, `indices`
/// holds the positions of `cases` in the input slice, and `cases` holds the cloned
/// cases in batch order — so `cases[i] == input[indices[i]]`. [`Batch::default`] is
/// the empty batch (all indices `0`, no cases); the derive exists so the permissive
/// `serde(default)` container attribute can fill in a key a payload omitted.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Batch {
    /// The 0-based index of the epoch this batch belongs to.
    pub epoch: usize,
    /// The 0-based index of this batch within its epoch.
    pub index: usize,
    /// The 1-based position of this batch across the whole plan.
    pub step: usize,
    /// The cases of this batch, cloned in batch order.
    pub cases: Vec<Case>,
    /// The positions of [`Batch::cases`] in the input slice, in batch order.
    pub indices: Vec<usize>,
}

impl Batch {
    /// The number of cases in this batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cases.len()
    }

    /// Whether this batch holds no cases.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A case with the given identifier.
    fn case(id: &str) -> Case {
        let mut inputs = std::collections::BTreeMap::new();
        inputs.insert(
            "question".to_owned(),
            serde_json::Value::String(id.to_owned()),
        );
        let mut label = std::collections::BTreeMap::new();
        label.insert(
            "answer".to_owned(),
            serde_json::Value::String(id.to_owned()),
        );
        Case::new(inputs, label).unwrap().with_case_id(id)
    }

    /// A list of `count` cases with the ids `0..count`.
    fn cases(count: usize) -> Vec<Case> {
        (0..count).map(|index| case(&index.to_string())).collect()
    }

    /// The batch sizes of a plan, all epochs flattened.
    fn sizes(plan: &[Epoch]) -> Vec<usize> {
        plan.iter()
            .flat_map(|epoch| epoch.batches.iter().map(Batch::len))
            .collect()
    }

    /// The case ids of a batch, in batch order.
    fn ids(batch: &Batch) -> Vec<String> {
        batch
            .cases
            .iter()
            .map(|case| case.case_id.clone())
            .collect()
    }

    #[test]
    fn batch_counts_keep_or_drop_the_partial_batch() {
        let input = cases(10);
        let keep = BatchScheduler::new(3).unwrap().plan(&input).unwrap();
        assert_eq!(keep.len(), 1);
        assert_eq!(sizes(&keep), [3, 3, 3, 1]);
        assert_eq!(keep[0].len(), 4);

        let drop = BatchScheduler::new(3)
            .unwrap()
            .with_drop_last()
            .plan(&input)
            .unwrap();
        assert_eq!(sizes(&drop), [3, 3, 3]);
        assert_eq!(drop[0].len(), 3);
        assert_eq!(sizes(&drop).iter().sum::<usize>(), 9);
    }

    #[test]
    fn an_exact_divisor_has_no_partial_batch() {
        let input = cases(9);
        let kept = BatchScheduler::new(3).unwrap().plan(&input).unwrap();
        assert_eq!(sizes(&kept), [3, 3, 3]);
        let dropped = BatchScheduler::new(3)
            .unwrap()
            .with_drop_last()
            .plan(&input)
            .unwrap();
        assert_eq!(sizes(&dropped), [3, 3, 3]);
        assert_eq!(kept, dropped, "an exact divisor needs no dropping");
    }

    #[test]
    fn fewer_cases_than_the_batch_size_fit_in_one_batch() {
        let input = cases(2);
        let plan = BatchScheduler::new(5).unwrap().plan(&input).unwrap();
        assert_eq!(sizes(&plan), [2]);
        assert_eq!(plan[0].batches[0].indices, [0, 1]);

        // ... unless `drop_last` drops the only batch there is.
        let dropped = BatchScheduler::new(5)
            .unwrap()
            .with_drop_last()
            .plan(&input)
            .unwrap();
        assert!(dropped[0].is_empty());
    }

    #[test]
    fn an_empty_case_list_plans_empty_epochs() {
        let plan = BatchScheduler::new(4)
            .unwrap()
            .with_epochs(2)
            .plan(&[])
            .unwrap();
        assert_eq!(plan.len(), 2);
        for (position, epoch) in plan.iter().enumerate() {
            assert!(epoch.is_empty());
            assert_eq!(epoch.len(), 0);
            assert_eq!(epoch.index, position);
        }
        assert_eq!(BatchScheduler::new(4).unwrap().batches_per_epoch(0), 0);
    }

    #[test]
    fn multiple_epochs_repeat_the_batch_layout() {
        let input = cases(7);
        let plan = BatchScheduler::new(3)
            .unwrap()
            .with_epochs(3)
            .plan(&input)
            .unwrap();
        assert_eq!(plan.len(), 3);
        assert_eq!(plan.iter().map(|e| e.index).collect::<Vec<_>>(), [0, 1, 2]);
        for epoch in &plan {
            assert_eq!(
                epoch.batches.iter().map(|b| b.index).collect::<Vec<_>>(),
                [0, 1, 2]
            );
            assert_eq!(sizes(std::slice::from_ref(epoch)), [3, 3, 1]);
        }
    }

    #[test]
    fn global_steps_number_the_whole_plan() {
        let input = cases(5);
        let plan = BatchScheduler::new(2)
            .unwrap()
            .with_epochs(3)
            .plan(&input)
            .unwrap();
        let steps: Vec<usize> = plan
            .iter()
            .flat_map(|epoch| epoch.batches.iter().map(|batch| batch.step))
            .collect();
        assert_eq!(steps, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn indices_and_cases_agree_with_the_input() {
        let input = cases(6);
        let plan = BatchScheduler::new(4)
            .unwrap()
            .with_epochs(2)
            .with_shuffle(31)
            .plan(&input)
            .unwrap();
        for batch in plan.iter().flat_map(|epoch| &epoch.batches) {
            assert_eq!(batch.cases.len(), batch.indices.len());
            for (case, index) in batch.cases.iter().zip(&batch.indices) {
                assert_eq!(case, &input[*index]);
            }
        }
    }

    #[test]
    fn unshuffled_epochs_keep_the_input_order() {
        let input = cases(5);
        let plan = BatchScheduler::new(2)
            .unwrap()
            .with_epochs(3)
            .plan(&input)
            .unwrap();
        for epoch in &plan {
            let plan_ids: Vec<String> = epoch.batches.iter().flat_map(ids).collect();
            assert_eq!(plan_ids, ["0", "1", "2", "3", "4"]);
            assert_eq!(
                epoch
                    .batches
                    .iter()
                    .flat_map(|batch| batch.indices.clone())
                    .collect::<Vec<_>>(),
                [0, 1, 2, 3, 4]
            );
        }
        assert_eq!(plan[0].batches[0].epoch, 0);
        assert_eq!(plan[2].batches[0].epoch, 2);
    }

    #[test]
    fn shuffled_plans_are_reproducible_and_epoch_specific() {
        let input = cases(12);
        let scheduler = BatchScheduler::new(5)
            .unwrap()
            .with_epochs(4)
            .with_shuffle(77);
        let first = scheduler.plan(&input).unwrap();
        assert_eq!(first, scheduler.plan(&input).unwrap());

        // Every epoch is a permutation of the input, and at least one epoch
        // differs from the input order (a shuffle that never reorders is a bug).
        let mut reordered = false;
        for epoch in &first {
            let mut plan_ids: Vec<usize> = epoch
                .batches
                .iter()
                .flat_map(|batch| batch.indices.clone())
                .collect();
            assert_eq!(plan_ids.len(), input.len());
            plan_ids.sort_unstable();
            assert_eq!(plan_ids, (0..input.len()).collect::<Vec<usize>>());
            reordered |= epoch
                .batches
                .iter()
                .flat_map(|batch| batch.indices.clone())
                .ne(0..input.len());
        }
        assert!(
            reordered,
            "a shuffled plan must differ from the input order"
        );

        // A different seed shuffles differently, and a different epoch within one
        // plan is seeded differently too.
        let other = BatchScheduler::new(5)
            .unwrap()
            .with_epochs(4)
            .with_shuffle(78)
            .plan(&input)
            .unwrap();
        assert_ne!(first, other);
        let orders: Vec<Vec<usize>> = first
            .iter()
            .map(|epoch| {
                epoch
                    .batches
                    .iter()
                    .flat_map(|batch| batch.indices.clone())
                    .collect()
            })
            .collect();
        assert!(
            orders.windows(2).any(|pair| pair[0] != pair[1]),
            "consecutive epochs must be able to shuffle differently"
        );
    }

    #[test]
    fn shuffled_epochs_keep_the_batch_layout() {
        let input = cases(11);
        let plan = BatchScheduler::new(3)
            .unwrap()
            .with_epochs(2)
            .with_shuffle(5)
            .plan(&input)
            .unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(sizes(&plan), [3, 3, 3, 2, 3, 3, 3, 2]);
    }

    #[test]
    fn new_rejects_a_zero_batch_size() {
        let error = BatchScheduler::new(0).unwrap_err();
        assert!(error.to_string().contains("batch_size"), "{error}");
    }

    #[test]
    fn plan_rejects_zero_epochs_and_a_deserialized_zero_batch_size() {
        let input = cases(3);
        let error = BatchScheduler::new(2)
            .unwrap()
            .with_epochs(0)
            .plan(&input)
            .unwrap_err();
        assert!(error.to_string().contains("epochs"), "{error}");

        let broken: BatchScheduler =
            serde_json::from_value(serde_json::json!({"batch_size": 0})).unwrap();
        assert_eq!(broken.batch_size(), 0);
        assert_eq!(broken.batches_per_epoch(10), 0);
        let error = broken.plan(&input).unwrap_err();
        assert!(error.to_string().contains("batch_size"), "{error}");
    }

    #[test]
    fn builders_round_trip_their_readers() {
        let scheduler = BatchScheduler::new(4)
            .unwrap()
            .with_epochs(2)
            .with_shuffle(9)
            .with_drop_last();
        assert_eq!(scheduler.batch_size(), 4);
        assert_eq!(scheduler.epochs(), 2);
        assert_eq!(scheduler.shuffle_seed(), Some(9));
        assert!(scheduler.drop_last());
        assert_eq!(scheduler.batches_per_epoch(10), 2);
        assert_eq!(scheduler.batches_per_epoch(0), 0);
        assert!(!scheduler.clone().without_drop_last().drop_last());
        assert_eq!(BatchScheduler::new(3).unwrap().shuffle_seed(), None);
    }

    #[test]
    fn scheduler_round_trips_through_json() {
        let scheduler = BatchScheduler::new(8)
            .unwrap()
            .with_epochs(3)
            .with_drop_last();
        let value = serde_json::to_value(&scheduler).unwrap();
        assert_eq!(value.as_object().map(serde_json::Map::len), Some(4));
        assert_eq!(value["batch_size"], serde_json::json!(8));
        assert_eq!(value["epochs"], serde_json::json!(3));
        assert_eq!(value["shuffle_seed"], serde_json::Value::Null);
        assert_eq!(value["drop_last"], serde_json::json!(true));

        let restored: BatchScheduler = serde_json::from_value(value).unwrap();
        assert_eq!(restored, scheduler);

        let shuffled = scheduler.clone().with_shuffle(12);
        let restored: BatchScheduler =
            serde_json::from_value(serde_json::to_value(&shuffled).unwrap()).unwrap();
        assert_eq!(restored.shuffle_seed(), Some(12));

        // Missing keys fall back to `Default`, unknown keys are ignored.
        let defaulted: BatchScheduler =
            serde_json::from_value(serde_json::json!({"epochs": 4, "unknown": 1})).unwrap();
        assert_eq!(defaulted, BatchScheduler::default().with_epochs(4));
        assert_eq!(
            serde_json::from_str::<BatchScheduler>("{}").unwrap(),
            BatchScheduler::default()
        );
    }

    #[test]
    fn plan_and_epochs_round_trip_through_json() {
        let input = cases(5);
        let plan = BatchScheduler::new(2)
            .unwrap()
            .with_epochs(2)
            .with_shuffle(3)
            .plan(&input)
            .unwrap();
        let restored: Vec<Epoch> =
            serde_json::from_value(serde_json::to_value(&plan).unwrap()).unwrap();
        assert_eq!(restored, plan);

        let batch = &plan[0].batches[0];
        assert_eq!(batch.epoch, 0);
        assert_eq!(batch.index, 0);
        assert_eq!(batch.step, 1);
        assert_eq!(batch.indices.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn default_matches_a_single_case_configuration() {
        let scheduler = BatchScheduler::default();
        assert_eq!(scheduler.batch_size(), 1);
        assert_eq!(scheduler.epochs(), DEFAULT_EPOCHS);
        assert_eq!(DEFAULT_EPOCHS, 1);
        assert_eq!(scheduler.shuffle_seed(), None);
        assert!(!scheduler.drop_last());

        let input = cases(3);
        let plan = scheduler.plan(&input).unwrap();
        assert_eq!(sizes(&plan), [1, 1, 1]);
        assert_eq!(plan.len(), 1);
    }
}
