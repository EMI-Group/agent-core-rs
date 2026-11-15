//! Dataset sampling, splitting, and batch scheduling over an in-memory case list.
//!
//! Ports the batch-scheduling and sampling layer the Python tune package keeps
//! above `dataset/case_loader.py`: seeded samplers (sequential, random subsample,
//! stratified by label) and an epoch/batch plan a trainer can iterate.
//!
//! [`SequentialSampler`] keeps the first `count` cases in input order,
//! [`RandomSampler`] takes `count` cases from a seeded Fisher-Yates permutation,
//! and [`StratifiedSampler`] allocates `count` proportionally over strata keyed by
//! the case label (or a caller-supplied function) with the largest-remainder
//! method. All three clamp a count larger than the case list, treat `count == 0` as
//! an empty sample, and are pure functions of `(cases, configuration, seed)`.
//!
//! [`BatchScheduler`] turns a case list into a `Vec<`[`Epoch`]`>` of [`Batch`]es:
//! a batch size, an epoch count, an optional shuffle seed, and a `drop_last` flag
//! produce a deterministic plan whose every epoch is a permutation of the input
//! and whose batches carry both the cloned cases and their input positions.
//!
//! [`CaseSplit`] tags which partition a case set or a score came from, and
//! [`SplitDataset`] bundles a case list with that tag, so a train/validation cut
//! keeps its provenance through evaluation instead of degrading to a bare
//! `list[Case]`. A [`SplitDataset`] can sample itself through
//! [`SplitDataset::sample`] and plan itself through [`SplitDataset::batches`].
//! Everything here is re-exported at the crate's `tune` root.

mod batch;
mod sampler;
mod split;

pub use batch::{Batch, BatchScheduler, Epoch, DEFAULT_EPOCHS};
pub use sampler::{CaseSampler, RandomSampler, SequentialSampler, StratifiedSampler};
pub use split::{CaseSplit, SplitDataset};
