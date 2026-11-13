//! Configuration of the single-harness optimisation loops.
//!
//! Ported from the request fields Python's `iterative.py` / `continuous.py`
//! read (`max_epochs`, `max_iterations`, the per-target margin, the retention
//! allowance, patience and the protected-case list), reduced to data: no live
//! models, clients, or environment lookups. Environment loading happens in the
//! host binary, never here.
//!
//! Every field is `snake_case` and defaulted, so a partial YAML document yields
//! the Python defaults for the missing keys; unknown keys are retained in
//! [`SingleHarnessConfig::extra`].

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use super::state::PromotionPolicyMode;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Field separator of the fingerprint payload (unit separator).
const FNV_SEPARATOR: u8 = 0x1f;

/// Feeds `bytes` into a running FNV-1a 64-bit hash.
fn fnv1a_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Configuration of one single-harness optimisation run.
///
/// The default mirrors Python's continuous policy: five epochs, ten iterations,
/// batches of five cases, a five-percent improvement margin and a five-percent
/// regression allowance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SingleHarnessConfig {
    /// Promotion policy; the continuous two-stage policy is the Python default.
    pub mode: PromotionPolicyMode,
    /// Maximum number of epochs (batches) the run may process.
    pub max_epochs: u32,
    /// Hard cap on attempted batches, shared by both policies.
    pub max_iterations: u32,
    /// How many cases one batch covers.
    pub batch_size: usize,
    /// Margin a case must close before it counts as satisfied (`0.05`).
    pub min_improvement: f64,
    /// Per-case loss a candidate may take without failing retention (`0.05`).
    pub max_regression: f64,
    /// Consecutive non-promotions that declare convergence.
    pub patience: u32,
    /// Optional score that ends the run once reached.
    pub target_score: Option<f64>,
    /// Attempts one batch may spend under the continuous policy.
    pub max_attempts_per_batch: u32,
    /// Candidate attempts the whole run may spend under the continuous policy.
    pub max_trials: u32,
    /// Whether the legacy policy promotes a candidate that simply beats the
    /// incumbent's overall score, even when the strict decision rejects it.
    ///
    /// The continuous policy ignores this flag: its promotion is gated by the
    /// two-stage decision only.
    pub promote_on_best_found: bool,
    /// Case ids that must never regress.
    pub protected_cases: Vec<String>,
    /// Signature of the evaluation protocol the scores are produced with.
    pub evaluation_protocol_signature: String,
    /// Unknown configuration keys, retained for forward compatibility.
    pub extra: Value,
}

impl Default for SingleHarnessConfig {
    fn default() -> Self {
        Self {
            mode: PromotionPolicyMode::ContinuousTwoStageV2,
            max_epochs: 5,
            max_iterations: 10,
            batch_size: 5,
            min_improvement: 0.05,
            max_regression: 0.05,
            patience: 2,
            target_score: None,
            max_attempts_per_batch: 2,
            max_trials: 8,
            promote_on_best_found: true,
            protected_cases: Vec::new(),
            evaluation_protocol_signature: String::from("v1"),
            extra: Value::Null,
        }
    }
}

impl SingleHarnessConfig {
    /// Validates the configuration.
    ///
    /// Rules: the batch must hold at least one case, the run must allow at
    /// least one iteration, and both margins must be finite and inside
    /// `0.0..=1.0`. Other zero values (`max_epochs`, `max_attempts_per_batch`,
    /// `max_trials`, `patience`) are permitted and simply bound their loop; a
    /// zero patience disables the convergence stop.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] for a zero `batch_size` or `max_iterations`,
    /// and for a non-finite or out-of-range `min_improvement` or
    /// `max_regression`.
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(Error::config(
                "single-harness batch_size must be greater than zero",
            ));
        }
        if self.max_iterations == 0 {
            return Err(Error::config(
                "single-harness max_iterations must be greater than zero",
            ));
        }
        for (name, value) in [
            ("min_improvement", self.min_improvement),
            ("max_regression", self.max_regression),
        ] {
            if !value.is_finite() {
                return Err(Error::config(format!(
                    "single-harness {name} must be finite, found {value}"
                )));
            }
            if !(0.0..=1.0).contains(&value) {
                return Err(Error::config(format!(
                    "single-harness {name} must be inside 0.0..=1.0, found {value}"
                )));
            }
        }
        Ok(())
    }

    /// Returns the deterministic fingerprint of a run over `dataset`.
    ///
    /// The payload is the dataset ids in the given order, the evaluation
    /// protocol signature, and the promotion-policy token, each terminated by a
    /// unit separator and hashed with 64-bit FNV-1a, rendered as 16 lowercase
    /// hex digits.
    ///
    /// The hash is stable across runs, processes, platforms and release
    /// profiles: it uses no randomised hasher, no clock and no allocation
    /// order. Two configurations with the same dataset, protocol signature and
    /// mode therefore share a fingerprint, which is exactly what
    /// [`super::state::SingleHarnessState::resume_from`] demands before it
    /// accepts persisted scores.
    #[must_use]
    pub fn fingerprint(&self, dataset: &[String]) -> String {
        let mut hash = FNV_OFFSET_BASIS;
        for case_id in dataset {
            hash = fnv1a_update(hash, case_id.as_bytes());
            hash = fnv1a_update(hash, &[FNV_SEPARATOR]);
        }
        hash = fnv1a_update(hash, self.evaluation_protocol_signature.as_bytes());
        hash = fnv1a_update(hash, &[FNV_SEPARATOR]);
        hash = fnv1a_update(hash, self.mode.token().as_bytes());
        format!("{hash:016x}")
    }

    /// Returns the protected case ids as a set, for fast membership checks.
    #[must_use]
    pub fn protected_case_set(&self) -> BTreeSet<&str> {
        self.protected_cases.iter().map(String::as_str).collect()
    }
}
