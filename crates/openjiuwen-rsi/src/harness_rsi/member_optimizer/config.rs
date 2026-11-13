//! Configuration for one team-member optimisation pass.
//!
//! Python's `MemberOptimizer` reads its limits from the auto-harness session
//! configuration (`max_roles_per_run` beam width, the single-harness
//! `max_actions_per_plan` cap, repair rounds, and the attribution confidence
//! floor). The Rust port keeps those knobs as one plain serde document so a
//! caller can carry them in a session config, a YAML file, or a test fixture.
//!
//! Nothing here touches the model runtime: the LLM attributors, planners and
//! executors stay behind the traits in [`super::selection`],
//! [`super::plan`], and [`super::optimizer`].
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
/// Python's default beam width (`max_roles_per_run`).
pub const DEFAULT_MAX_ROLES_PER_RUN: usize = 2;
/// Python's default single-harness action cap; `3` actions per plan.
pub const DEFAULT_MAX_ACTIONS_PER_PLAN: usize = 3;
/// Python's default number of repair rounds after a failed verification.
pub const DEFAULT_MAX_REPAIR_ROUNDS: usize = 1;
/// Limits for one member-optimisation pass.
///
/// Every field is `snake_case`, defaults to the Python value, and tolerates
/// unknown keys so a configuration file shared with the Python implementation
/// keeps loading.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberOptimizerConfig {
    /// How many roles one run optimises (the deterministic selection beam width).
    pub max_roles_per_run: usize,
    /// Cap on actions in a single plan; `0` means unlimited.
    pub max_actions_per_plan: usize,
    /// How many repair rounds follow a repairable verification failure; `0`
    /// disables repair.
    pub max_repair_rounds: usize,
    /// Attribution confidence floor; targets below it are never selected.
    pub min_confidence: f64,
    /// When `true`, a verification failure that cannot be repaired fails the
    /// whole call instead of returning a `Failed` artifact.
    pub fail_on_unverifiable: bool,
    /// Caller annotations, carried through to the artifact metadata untouched.
    pub extra: Value,
}
impl Default for MemberOptimizerConfig {
    /// Returns the Python defaults.
    fn default() -> Self {
        Self {
            max_roles_per_run: DEFAULT_MAX_ROLES_PER_RUN,
            max_actions_per_plan: DEFAULT_MAX_ACTIONS_PER_PLAN,
            max_repair_rounds: DEFAULT_MAX_REPAIR_ROUNDS,
            min_confidence: 0.0,
            fail_on_unverifiable: false,
            extra: Value::Null,
        }
    }
}
impl MemberOptimizerConfig {
    /// Validates the limits before a run starts.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Config`] when
    /// `max_roles_per_run` is `0`, when `min_confidence` is not finite or falls
    /// outside `0.0..=1.0`, or when `max_repair_rounds`/`max_actions_per_plan`
    /// cannot be represented.
    pub fn validate(&self) -> Result<()> {
        if self.max_roles_per_run == 0 {
            return Err(Error::config(
                "max_roles_per_run must be at least 1 to select a member target",
            ));
        }
        if !self.min_confidence.is_finite() {
            return Err(Error::config("min_confidence must be a finite number"));
        }
        if !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(Error::config(format!(
                "min_confidence must be within 0.0..=1.0, got {}",
                self.min_confidence
            )));
        }
        Ok(())
    }
    /// Effective action cap, with `0` mapped to "unlimited".
    ///
    /// Planners use `0` as the unrestricted sentinel, matching
    /// [`DeterministicActionPlanner`](super::plan::DeterministicActionPlanner).
    #[must_use]
    pub fn max_actions_cap(&self) -> usize {
        if self.max_actions_per_plan == 0 {
            usize::MAX
        } else {
            self.max_actions_per_plan
        }
    }
}
