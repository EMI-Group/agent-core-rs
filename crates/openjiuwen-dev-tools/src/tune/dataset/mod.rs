//! Dataset sampling, splitting, and batch scheduling over an in-memory case list.
//!
//! Ports the batch-scheduling and sampling layer the Python tune package keeps
//! above `dataset/case_loader.py`: seeded samplers (random subsample, stratified
//! by label, sequential) and an epoch/batch plan a trainer can iterate.
//!
//! [`CaseSplit`] tags which partition a case set or a score came from, and
//! [`SplitDataset`] bundles a case list with that tag, so a train/validation cut
//! keeps its provenance through evaluation instead of degrading to a bare
//! `list[Case]`. Both are re-exported at the crate's `tune` root.

mod split;

pub use split::{CaseSplit, SplitDataset};
