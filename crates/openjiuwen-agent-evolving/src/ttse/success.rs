//! Success detection over a recorded [`Trajectory`](crate::trajectory::Trajectory).
//!
//! Ports the offline intent of Python `openjiuwen.agent_evolving.ttse.success`:
//! a run counts as successful when its recorded outcome says so, or when the
//! outcome is undecided but the run shows enough tool activity and no error
//! step. The detector is pure and deterministic — no model, no clock.
//!
//! [`SuccessDetector::detect`] returns a [`SuccessAssessment`] whose
//! [`evidence`](SuccessAssessment::evidence) line always starts with the
//! recorded outcome and carries the tool-call and error counts, so a caller can
//! log or store the decision without re-reading the trajectory.

use crate::trajectory::{StepKind, Trajectory, TrajectoryOutcome};
use crate::ttse::config::TtseConfig;
use crate::ttse::models::{truncate_chars, SuccessAssessment};

/// Longest evidence line the detector produces, in characters.
const MAX_EVIDENCE_CHARS: usize = 200;
/// Longest first-error excerpt appended to a failure's evidence, in characters.
const MAX_ERROR_EXCERPT_CHARS: usize = 80;

/// Decides whether one trajectory counts as a successful experience.
///
/// The recorded [`Trajectory::outcome`] wins when it is decided: `Success` scores
/// `1.0`, `Failure` scores `0.0`. An `Unknown` outcome is decided by activity
/// instead — no error step and at least [`min_tool_calls`](Self::min_tool_calls)
/// tool calls score `1.0`, everything else scores `0.0`.
#[derive(Debug, Clone, PartialEq)]
pub struct SuccessDetector {
    /// How many tool calls an undecided run needs to count as a success.
    min_tool_calls: usize,
    /// The score a run must reach to be reported as successful.
    success_threshold: f64,
}

impl SuccessDetector {
    /// Builds a detector, clamping the threshold into `0.0..=1.0`.
    ///
    /// A non-finite threshold (for example `f64::NAN`) becomes `1.0`, so a
    /// broken config cannot silently accept every run.
    #[must_use]
    pub fn new(min_tool_calls: usize, success_threshold: f64) -> Self {
        let success_threshold = if success_threshold.is_finite() {
            success_threshold.clamp(0.0, 1.0)
        } else {
            1.0
        };
        Self {
            min_tool_calls,
            success_threshold,
        }
    }

    /// Builds a detector from the TTSE tunables.
    ///
    /// Uses [`TtseConfig::detect_min_tool_calls`] and
    /// [`TtseConfig::success_threshold`].
    #[must_use]
    pub fn from_config(config: &TtseConfig) -> Self {
        Self::new(config.detect_min_tool_calls, config.success_threshold)
    }

    /// The minimum number of tool calls an undecided run needs.
    #[must_use]
    pub fn min_tool_calls(&self) -> usize {
        self.min_tool_calls
    }

    /// The score a run must reach to be reported as successful.
    #[must_use]
    pub fn success_threshold(&self) -> f64 {
        self.success_threshold
    }

    /// Assesses one recorded trajectory.
    ///
    /// The assessment is a pure function of the recorded values: the same
    /// trajectory always yields the same assessment.
    #[must_use]
    pub fn detect(&self, trajectory: &Trajectory) -> SuccessAssessment {
        let tool_calls = trajectory
            .steps()
            .iter()
            .filter(|step| step.kind == StepKind::ToolCall)
            .count();
        let errors = trajectory.error_count();
        let (base_success, score) = match trajectory.outcome {
            TrajectoryOutcome::Success => (true, 1.0),
            TrajectoryOutcome::Failure => (false, 0.0),
            TrajectoryOutcome::Unknown => {
                let decided = errors == 0 && tool_calls >= self.min_tool_calls;
                (decided, if decided { 1.0 } else { 0.0 })
            }
        };

        let mut evidence = format!(
            "outcome={} tool_calls={tool_calls} errors={errors}",
            trajectory.outcome.as_str()
        );
        if trajectory.outcome == TrajectoryOutcome::Failure {
            if let Some(first_error) = first_error_excerpt(trajectory) {
                evidence.push_str(" first_error=");
                evidence.push_str(&first_error);
            }
        }
        let evidence = truncate_chars(&evidence, MAX_EVIDENCE_CHARS);

        SuccessAssessment {
            successful: base_success && score >= self.success_threshold,
            score,
            evidence,
        }
    }
}

/// The excerpt of the first step that records an error, when one exists.
///
/// Steps whose content is blank fall through to the next error step; `None`
/// means there is no usable error excerpt to report.
fn first_error_excerpt(trajectory: &Trajectory) -> Option<String> {
    trajectory
        .steps()
        .iter()
        .filter(|step| step.is_error())
        .map(|step| step.excerpt(MAX_ERROR_EXCERPT_CHARS))
        .find(|excerpt| !excerpt.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{StepStatus, TrajectoryStep};

    fn step(kind: StepKind, content: &str) -> TrajectoryStep {
        TrajectoryStep::new(kind, content)
    }

    fn failing_step(content: &str) -> TrajectoryStep {
        step(StepKind::ToolResult, content).with_status(StepStatus::Error)
    }

    fn trajectory(outcome: TrajectoryOutcome, steps: Vec<TrajectoryStep>) -> Trajectory {
        let mut trajectory = Trajectory::new("t-1").expect("valid trajectory id");
        trajectory.outcome = outcome;
        trajectory.extend(steps);
        trajectory
    }

    #[test]
    fn success_outcome_scores_full_and_ignores_errors() {
        let detector = SuccessDetector::new(5, 0.999);
        let recorded = trajectory(
            TrajectoryOutcome::Success,
            vec![
                step(StepKind::ToolCall, "run cargo test"),
                failing_step("a retried command failed"),
            ],
        );
        let assessment = detector.detect(&recorded);
        assert!(assessment.successful);
        assert!((assessment.score - 1.0).abs() < f64::EPSILON);
        assert!(assessment.evidence.starts_with("outcome=success"));
        assert!(assessment.evidence.contains("tool_calls=1"));
        assert!(assessment.evidence.contains("errors=1"));
        assert!(!assessment.evidence.contains("first_error="));
    }

    #[test]
    fn failure_outcome_reports_the_first_error_excerpt() {
        let detector = SuccessDetector::new(0, 0.999);
        let recorded = trajectory(
            TrajectoryOutcome::Failure,
            vec![
                step(StepKind::ToolCall, "run cargo build"),
                failing_step("command not found: catt"),
                failing_step("second error"),
            ],
        );
        let assessment = detector.detect(&recorded);
        assert!(!assessment.successful);
        assert!((assessment.score - 0.0).abs() < f64::EPSILON);
        assert!(assessment.evidence.starts_with("outcome=failure"));
        assert!(assessment.evidence.contains("tool_calls=1"));
        assert!(assessment.evidence.contains("errors=2"));
        assert!(assessment
            .evidence
            .contains("first_error=command not found: catt"));
        assert!(!assessment.evidence.contains("second error"));
    }

    #[test]
    fn unknown_with_no_errors_and_enough_tool_calls_succeeds() {
        let detector = SuccessDetector::new(2, 0.999);
        let recorded = trajectory(
            TrajectoryOutcome::Unknown,
            vec![
                step(StepKind::ToolCall, "run rg"),
                step(StepKind::ToolResult, "matches"),
                step(StepKind::ToolCall, "run cargo test"),
                step(StepKind::Message, "done"),
            ],
        );
        let assessment = detector.detect(&recorded);
        assert!(assessment.successful);
        assert!((assessment.score - 1.0).abs() < f64::EPSILON);
        assert!(assessment.evidence.starts_with("outcome=unknown"));
        assert!(assessment.evidence.contains("tool_calls=2"));
        assert!(assessment.evidence.contains("errors=0"));
    }

    #[test]
    fn unknown_with_an_error_step_fails() {
        let detector = SuccessDetector::new(1, 0.999);
        let recorded = trajectory(
            TrajectoryOutcome::Unknown,
            vec![
                step(StepKind::ToolCall, "run cargo test"),
                failing_step("test failed"),
            ],
        );
        let assessment = detector.detect(&recorded);
        assert!(!assessment.successful);
        assert!((assessment.score - 0.0).abs() < f64::EPSILON);
        assert!(assessment.evidence.contains("errors=1"));
        assert!(!assessment.evidence.contains("first_error="));
    }

    #[test]
    fn unknown_with_too_few_tool_calls_fails() {
        let detector = SuccessDetector::new(5, 0.999);
        let recorded = trajectory(
            TrajectoryOutcome::Unknown,
            vec![
                step(StepKind::ToolCall, "run rg"),
                step(StepKind::Message, "chatting"),
            ],
        );
        let assessment = detector.detect(&recorded);
        assert!(!assessment.successful);
        assert!((assessment.score - 0.0).abs() < f64::EPSILON);
        assert!(assessment.evidence.contains("tool_calls=1"));
    }

    #[test]
    fn evidence_is_bounded_and_the_error_excerpt_is_capped() {
        let detector = SuccessDetector::new(0, 0.5);
        let recorded = trajectory(
            TrajectoryOutcome::Failure,
            vec![failing_step(&"x".repeat(300))],
        );
        let assessment = detector.detect(&recorded);
        assert!(assessment.evidence.chars().count() <= MAX_EVIDENCE_CHARS);
        assert!(
            !assessment
                .evidence
                .contains(&"x".repeat(MAX_ERROR_EXCERPT_CHARS + 1)),
            "the error excerpt must be capped at {MAX_ERROR_EXCERPT_CHARS} chars: {}",
            assessment.evidence
        );
        assert!(assessment.evidence.contains('…'));
    }

    #[test]
    fn detection_is_repeatable_and_the_config_mirrors_its_values() {
        let config = TtseConfig::default();
        let detector = SuccessDetector::from_config(&config);
        assert_eq!(detector.min_tool_calls(), config.detect_min_tool_calls);
        assert!((detector.success_threshold() - config.success_threshold).abs() < f64::EPSILON);

        let recorded = trajectory(
            TrajectoryOutcome::Unknown,
            vec![step(StepKind::ToolCall, "run rg")],
        );
        assert_eq!(detector.detect(&recorded), detector.detect(&recorded));

        assert!((SuccessDetector::new(0, 2.0).success_threshold() - 1.0).abs() < f64::EPSILON);
        assert!((SuccessDetector::new(0, -0.5).success_threshold() - 0.0).abs() < f64::EPSILON);
        assert!((SuccessDetector::new(0, f64::NAN).success_threshold() - 1.0).abs() < f64::EPSILON);
        assert!(
            (SuccessDetector::new(0, f64::INFINITY).success_threshold() - 1.0).abs() < f64::EPSILON
        );

        let successful = detector.detect(&trajectory(TrajectoryOutcome::Success, Vec::new()));
        assert!(successful.successful);
    }
}
