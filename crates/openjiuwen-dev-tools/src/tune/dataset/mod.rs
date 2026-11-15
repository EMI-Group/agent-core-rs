//! Dataset sampling and batch scheduling over an in-memory case list.
//!
//! Ports the batch-scheduling and sampling layer the Python tune package keeps
//! above `dataset/case_loader.py`: seeded samplers (random subsample, stratified
//! by label, sequential) and an epoch/batch plan a trainer can iterate.
