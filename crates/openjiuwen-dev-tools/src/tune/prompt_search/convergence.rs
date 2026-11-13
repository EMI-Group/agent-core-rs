//! Convergence detection for the prompt-search optimization loop.
//!
//! Ports `convergence.py`. The loop stops when further iterations are unlikely
//! to help: a caller-supplied target reward is reached, no improvement greater
//! than `threshold` appeared over the last `window` iterations, or the variance
//! of the recent rewards flattened below `threshold²`.
//!
//! Python's detector accumulates rewards through `update`; here
//! [`ConvergenceDetector::check`] takes the whole reward series, so the detector
//! is a pure value and a caller can re-check a truncated history without owning
//! mutable state.

use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

/// The population variance of `values` (`/n`), `0.0` for fewer than two values.
///
/// Non-finite values are treated as `0.0`, so the variance is always finite.
#[must_use]
pub fn population_variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let count = values.len() as f64;
    let mut mean = 0.0;
    for value in values {
        mean += finite(*value);
    }
    mean /= count;
    let mut sum = 0.0;
    for value in values {
        let delta = finite(*value) - mean;
        sum += delta * delta;
    }
    sum / count
}

/// Replaces a non-finite value with `0.0`.
fn finite(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

/// The outcome of a convergence check.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ConvergenceState {
    /// Whether the loop should stop.
    pub converged: bool,
    /// Why it should stop, in human-readable form.
    pub reason: String,
    /// The best reward seen so far.
    pub best: f64,
    /// The mean of the last `window` rewards.
    pub moving_average: f64,
    /// The population variance of the last `window` rewards.
    pub variance: f64,
}

/// Detects when the optimization loop has converged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ConvergenceDetector {
    /// The improvement (or variance) floor below which the loop stops.
    threshold: f64,
    /// How many recent rewards form the plateau window.
    window: usize,
    /// An optional reward at which the loop stops immediately.
    target: Option<f64>,
}

impl Default for ConvergenceDetector {
    fn default() -> Self {
        Self {
            threshold: 0.01,
            window: 3,
            target: None,
        }
    }
}

impl ConvergenceDetector {
    /// Creates a detector.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `threshold` is non-finite or not positive, or
    /// when `window` is zero.
    pub fn new(threshold: f64, window: usize, target: Option<f64>) -> Result<Self> {
        if !threshold.is_finite() || threshold <= 0.0 {
            return Err(Error::validation(
                "ConvergenceDetector.threshold must be finite and positive",
            ));
        }
        if window == 0 {
            return Err(Error::validation(
                "ConvergenceDetector.window must be at least 1",
            ));
        }
        Ok(Self {
            threshold,
            window,
            target,
        })
    }

    /// Sets (or clears) the target reward.
    #[must_use]
    pub fn with_target(mut self, target: Option<f64>) -> Self {
        self.target = target;
        self
    }

    /// The configured improvement/variance threshold.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// The configured plateau window.
    #[must_use]
    pub fn window(&self) -> usize {
        self.window
    }

    /// The configured target reward.
    #[must_use]
    pub fn target(&self) -> Option<f64> {
        self.target
    }

    /// Checks the reward series, returning the state when the loop has
    /// converged and `None` otherwise.
    ///
    /// Rules are applied in Python's order of priority: target reached, then a
    /// plateau over the last `window` iterations (which needs one more reward to
    /// compare against), then a variance below `threshold²`. The last `window`
    /// rewards (or fewer) supply `moving_average` and `variance`; an empty input
    /// yields `None`.
    #[must_use]
    pub fn check(&self, rewards: &[f64]) -> Option<ConvergenceState> {
        if rewards.is_empty() {
            return None;
        }
        let mut best = f64::NEG_INFINITY;
        for reward in rewards {
            best = best.max(finite(*reward));
        }
        let recent = &rewards[rewards.len().saturating_sub(self.window)..];
        let moving_average =
            recent.iter().map(|reward| finite(*reward)).sum::<f64>() / recent.len() as f64;
        let variance = population_variance(recent);

        let mut state = ConvergenceState {
            converged: false,
            reason: String::new(),
            best,
            moving_average,
            variance,
        };

        if let Some(target) = self.target {
            if best >= target {
                state.converged = true;
                state.reason = format!("target reward {target:.3} reached");
                return Some(state);
            }
        }

        if rewards.len() > self.window {
            let prior = &rewards[..rewards.len() - self.window];
            let mut prior_best = f64::NEG_INFINITY;
            for reward in prior {
                prior_best = prior_best.max(finite(*reward));
            }
            if best - prior_best <= self.threshold {
                state.converged = true;
                state.reason = format!(
                    "no improvement > {:.3} over last {} iterations",
                    self.threshold, self.window
                );
                return Some(state);
            }
        }

        if recent.len() >= self.window && variance <= self.threshold * self.threshold {
            state.converged = true;
            state.reason = format!("reward variance {variance:.5} below plateau threshold");
            return Some(state);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn population_variance_edges() {
        assert_eq!(population_variance(&[]), 0.0);
        assert_eq!(population_variance(&[0.5]), 0.0);
        assert!((population_variance(&[0.0, 1.0]) - 0.25).abs() < 1e-12);
        assert!((population_variance(&[1.0, 1.0, 1.0]) - 0.0).abs() < 1e-12);
        assert_eq!(population_variance(&[f64::NAN, f64::NAN]), 0.0);
    }

    #[test]
    fn new_validates_its_configuration() {
        assert!(ConvergenceDetector::new(0.01, 3, None).is_ok());
        for threshold in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            let error = ConvergenceDetector::new(threshold, 3, None).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
        }
        let error = ConvergenceDetector::new(0.01, 0, None).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn default_configuration() {
        let detector = ConvergenceDetector::default();
        assert!((detector.threshold() - 0.01).abs() < 1e-12);
        assert_eq!(detector.window(), 3);
        assert_eq!(detector.target(), None);
        assert_eq!(
            ConvergenceDetector::default()
                .with_target(Some(0.9))
                .target(),
            Some(0.9)
        );
    }

    #[test]
    fn target_rule_wins_over_the_others() {
        let detector = ConvergenceDetector::new(0.01, 3, Some(0.8)).unwrap();
        let state = detector.check(&[0.8, 0.8, 0.8, 0.8]).unwrap();
        assert!(state.converged);
        assert_eq!(state.reason, "target reward 0.800 reached");
        assert!((state.best - 0.8).abs() < 1e-12);
    }

    #[test]
    fn plateau_rule_needs_window_plus_one_rewards() {
        let detector = ConvergenceDetector::new(0.01, 3, None).unwrap();

        // Only a full window: the variance rule may fire, the plateau rule may not.
        assert!(detector.check(&[0.5, 0.5]).is_none());
        assert!(detector.check(&[0.0, 1.0, 0.5]).is_none());
        let state = detector.check(&[0.5, 0.501, 0.5005]).unwrap();
        assert!(state.converged);
        assert!(state.variance <= 0.01 * 0.01);
        assert!(
            state.reason.starts_with("reward variance "),
            "{}",
            state.reason
        );

        // Four rewards with no improvement over the last three.
        let state = detector.check(&[0.40, 0.41, 0.41, 0.41]).unwrap();
        assert_eq!(
            state.reason,
            "no improvement > 0.010 over last 3 iterations"
        );
        assert!((state.best - 0.41).abs() < 1e-12);
        assert!((state.moving_average - 0.41).abs() < 1e-12);
    }

    #[test]
    fn variance_rule_uses_the_last_window() {
        let detector = ConvergenceDetector::new(0.1, 2, None).unwrap();
        // Prior best grows, so the plateau rule is skipped (len == window), and
        // the last two rewards are identical.
        let state = detector.check(&[0.0, 0.5, 0.5]).unwrap();
        assert_eq!(
            state.reason,
            "reward variance 0.00000 below plateau threshold"
        );
        assert!((state.moving_average - 0.5).abs() < 1e-12);
    }

    #[test]
    fn growth_does_not_converge() {
        let detector = ConvergenceDetector::new(0.01, 3, None).unwrap();
        assert!(detector.check(&[]).is_none());
        assert!(detector.check(&[0.1]).is_none());
        assert!(detector.check(&[0.1, 0.3]).is_none());
        assert!(detector.check(&[0.1, 0.3, 0.6]).is_none());
        assert!(detector.check(&[0.1, 0.3, 0.6, 0.9]).is_none());
    }

    #[test]
    fn state_serializes_with_python_field_names() {
        let state = ConvergenceState {
            converged: true,
            reason: "done".to_owned(),
            best: 0.9,
            moving_average: 0.85,
            variance: 0.0001,
        };
        let value = serde_json::to_value(&state).unwrap();
        for key in ["converged", "reason", "best", "moving_average", "variance"] {
            assert!(value.get(key).is_some(), "missing {key}: {value}");
        }
        let parsed: ConvergenceState = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, state);
    }
}
