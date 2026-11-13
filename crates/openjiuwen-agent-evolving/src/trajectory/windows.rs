//! Bounded views over a trajectory.
//!
//! Ports the windowing intent of Python
//! `openjiuwen.agent_evolving.trajectory.windows` to typed step slices: a
//! window is a contiguous run of [`TrajectoryStep`]s that fits a caller's
//! step budget, used for prompt injection and experience extraction without
//! replaying a whole recording.
//!
//! All ranges are positions in [`Trajectory::steps`]; `start` is inclusive
//! and `end` is exclusive. Every helper is deterministic and offline.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::model::{render_step_line, truncate_chars, Trajectory, TrajectoryStep};

/// A contiguous slice of a trajectory's steps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryWindow {
    /// Position of the first step (inclusive).
    #[serde(default)]
    pub start: usize,
    /// Position after the last step (exclusive).
    #[serde(default)]
    pub end: usize,
    /// The copied steps of the window, in trajectory order.
    #[serde(default)]
    pub steps: Vec<TrajectoryStep>,
}

impl TrajectoryWindow {
    /// Number of steps in the window.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the window holds no step.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// Splits `trajectory` into overlapping windows of at most `max_steps` steps.
///
/// Consecutive windows advance by `max_steps - overlap`, so `overlap` steps
/// are repeated at each boundary. `max_steps` must be positive and `overlap`
/// must be smaller than `max_steps`; both violations are
/// [`Error::validation`]. An empty trajectory yields no windows.
pub fn windows(
    trajectory: &Trajectory,
    max_steps: usize,
    overlap: usize,
) -> Result<Vec<TrajectoryWindow>> {
    if max_steps == 0 {
        return Err(Error::validation("windows: max_steps must be > 0"));
    }
    if overlap >= max_steps {
        return Err(Error::validation("windows: overlap must be < max_steps"));
    }
    let steps = trajectory.steps();
    if steps.is_empty() {
        return Ok(Vec::new());
    }
    let stride = max_steps - overlap;
    let mut windows = Vec::new();
    let mut start = 0usize;
    loop {
        let end = start.saturating_add(max_steps).min(steps.len());
        windows.push(TrajectoryWindow {
            start,
            end,
            steps: steps[start..end].to_vec(),
        });
        if end == steps.len() {
            break;
        }
        start += stride;
    }
    Ok(windows)
}

/// The window of `radius` steps on either side of `step_index`.
///
/// `step_index` is a position in [`Trajectory::steps`]; it must exist, or the
/// call fails with [`Error::validation`]. The window is clamped to the
/// trajectory bounds, so the result is never empty and `radius == 0` yields
/// exactly the single step at `step_index`.
pub fn window_around(
    trajectory: &Trajectory,
    step_index: usize,
    radius: usize,
) -> Result<TrajectoryWindow> {
    let steps = trajectory.steps();
    if step_index >= steps.len() {
        return Err(Error::validation(format!(
            "window_around: step index {step_index} out of bounds (len {})",
            steps.len()
        )));
    }
    let start = step_index.saturating_sub(radius);
    let end = step_index
        .saturating_add(radius)
        .saturating_add(1)
        .min(steps.len());
    Ok(TrajectoryWindow {
        start,
        end,
        steps: steps[start..end].to_vec(),
    })
}

/// Renders one line per step (`#index kind role: excerpt`) within `max_chars`.
///
/// Lines are added whole while they fit; when the next line would overflow,
/// the rendering stops and a trailing `…` marks the cut, so the result is
/// never longer than `max_chars` characters (and exactly `…` when
/// `max_chars == 1`). `max_chars == 0` yields an empty string. Each excerpt
/// is capped at 80 characters, and the rendering depends only on the recorded
/// steps, so repeated calls are identical.
#[must_use]
pub fn render_digest(trajectory: &Trajectory, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let lines: Vec<String> = trajectory.steps().iter().map(render_step_line).collect();
    let mut rendered = String::new();
    let mut used = 0usize;
    let mut truncated = false;
    for (position, line) in lines.iter().enumerate() {
        let separator = usize::from(position > 0);
        let line_chars = line.chars().count();
        // Reserve the ellipsis while more lines could follow this one.
        let reserve = usize::from(position + 1 < lines.len());
        if used + separator + line_chars + reserve > max_chars {
            truncated = true;
            break;
        }
        if separator == 1 {
            rendered.push('\n');
        }
        rendered.push_str(line);
        used += separator + line_chars;
    }
    if truncated {
        if rendered.is_empty() {
            // Not even the first line fits: keep a bounded excerpt of it.
            rendered = lines
                .first()
                .map_or_else(String::new, |line| truncate_chars(line, max_chars));
            if rendered.is_empty() {
                rendered.push('…');
            }
        } else if rendered.chars().count() < max_chars {
            rendered.push('…');
        } else {
            let mut cut: String = rendered.chars().take(max_chars - 1).collect();
            cut.push('…');
            rendered = cut;
        }
    }
    rendered
}

/// The first step whose status is [`super::model::StepStatus::Error`].
#[must_use]
pub fn first_error(trajectory: &Trajectory) -> Option<&TrajectoryStep> {
    trajectory.steps().iter().find(|step| step.is_error())
}

/// The last `n` steps (the whole trajectory when `n` exceeds its length).
#[must_use]
pub fn last_n(trajectory: &Trajectory, n: usize) -> &[TrajectoryStep] {
    let start = trajectory.len().saturating_sub(n);
    &trajectory.steps()[start..]
}

#[cfg(test)]
mod tests {
    use openjiuwen_common::error::Error;

    use super::*;
    use crate::trajectory::model::{StepKind, StepStatus, TrajectoryStep};

    fn trajectory_with(count: usize) -> Trajectory {
        let mut trajectory = Trajectory::new("t-1").expect("valid id");
        for position in 0..count {
            trajectory.push(TrajectoryStep::new(
                StepKind::Message,
                format!("step {position}"),
            ));
        }
        trajectory
    }

    #[test]
    fn windows_cover_the_trajectory_without_gaps() {
        let trajectory = trajectory_with(5);
        let split = windows(&trajectory, 2, 0).expect("valid windows");
        let ranges: Vec<(usize, usize)> = split
            .iter()
            .map(|window| (window.start, window.end))
            .collect();
        assert_eq!(ranges, vec![(0, 2), (2, 4), (4, 5)]);
        assert_eq!(split[2].len(), 1);
    }

    #[test]
    fn overlap_repeats_boundary_steps() {
        let trajectory = trajectory_with(4);
        let split = windows(&trajectory, 3, 1).expect("valid windows");
        let ranges: Vec<(usize, usize)> = split
            .iter()
            .map(|window| (window.start, window.end))
            .collect();
        assert_eq!(ranges, vec![(0, 3), (2, 4)]);
    }

    #[test]
    fn window_validation_and_bounds() {
        let empty = Trajectory::new("t-1").expect("valid id");
        assert!(matches!(windows(&empty, 0, 0), Err(Error::Validation(_))));
        assert!(matches!(windows(&empty, 1, 1), Err(Error::Validation(_))));
        assert_eq!(windows(&empty, 1, 0).expect("valid").len(), 0);
        assert!(window_around(&empty, 0, 1).is_err());

        let trajectory = trajectory_with(3);
        let around = window_around(&trajectory, 1, 1).expect("in bounds");
        assert_eq!((around.start, around.end), (0, 3));
        assert!(window_around(&trajectory, 3, 1).is_err());
    }

    #[test]
    fn digest_truncation_is_utf8_safe() {
        let mut trajectory = Trajectory::new("t-1").expect("valid id");
        trajectory
            .push(TrajectoryStep::new(StepKind::Message, "日本語のテキスト").with_role("user"));
        let rendered = render_digest(&trajectory, 10);
        assert_eq!(rendered.chars().count(), 10);
        assert!(rendered.ends_with('…'));
        assert_eq!(render_digest(&trajectory, 0), "");
        assert_eq!(render_digest(&trajectory, 1), "…");
        assert_eq!(
            render_digest(&trajectory, 1_000),
            render_digest(&trajectory, 1_000)
        );
    }

    #[test]
    fn error_and_tail_helpers() {
        let mut trajectory = trajectory_with(3);
        trajectory
            .push(TrajectoryStep::new(StepKind::ToolResult, "boom").with_status(StepStatus::Error));
        assert_eq!(first_error(&trajectory).map(|step| step.index), Some(3));
        assert_eq!(last_n(&trajectory, 2).len(), 2);
        assert_eq!(last_n(&trajectory, 0).len(), 0);
        assert_eq!(last_n(&trajectory, 99).len(), 4);
    }
}
