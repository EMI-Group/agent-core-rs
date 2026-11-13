//! Span projections over recorded steps.
//!
//! Ports the tool-pairing part of Python
//! `openjiuwen.agent_evolving.trajectory.spans` without the OTLP envelope: a
//! span is one [`StepKind::ToolCall`] step plus the first later
//! [`StepKind::ToolResult`] step that answers it.
//!
//! Ranges are *positions* in [`Trajectory::steps`] (`start` inclusive, `end`
//! exclusive when it points at a result). An unanswered call becomes an open
//! span: `end == start`, `open == true`, and the span's status is the call's
//! own status. A call whose id is absent or empty can never pair, so it is
//! always open.

use std::cmp::Reverse;

use serde::{Deserialize, Serialize};

use super::model::{StepKind, StepStatus, Trajectory};

/// One span: a tool call plus the result that answered it, if any.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrajectorySpan {
    /// Deterministic identifier, `span-{start}`.
    pub span_id: String,
    /// Tool name of the call, or `tool_call` when the step recorded none.
    pub name: String,
    /// The step kind that opened the span (always [`StepKind::ToolCall`]).
    pub kind: StepKind,
    /// Position of the call in [`Trajectory::steps`].
    pub start: usize,
    /// Position of the answering result, or [`start`](Self::start) when open.
    pub end: usize,
    /// The result's status for a closed span, the call's status for an open one.
    pub status: StepStatus,
    /// Marker for an unanswered call (`end == start`).
    pub open: bool,
}

impl Default for TrajectorySpan {
    fn default() -> Self {
        Self {
            span_id: String::new(),
            name: String::new(),
            kind: StepKind::Other,
            start: 0,
            end: 0,
            status: StepStatus::Ok,
            open: false,
        }
    }
}

/// Projects every tool-call step into a [`TrajectorySpan`], in step order.
///
/// Matching is positional: the first later `ToolResult` step whose non-empty
/// `tool_call_id` equals the call's id closes the span. Results recorded
/// *before* their call never pair, and one result can close at most the calls
/// that precede it.
#[must_use]
pub fn spans(trajectory: &Trajectory) -> Vec<TrajectorySpan> {
    let steps = trajectory.steps();
    let mut projected = Vec::new();
    for (position, step) in steps.iter().enumerate() {
        if step.kind != StepKind::ToolCall {
            continue;
        }
        let call_id = step
            .tool_call_id
            .as_deref()
            .filter(|call_id| !call_id.is_empty());
        let closed_at = call_id.and_then(|call_id| {
            steps
                .iter()
                .enumerate()
                .skip(position + 1)
                .find(|(_, candidate)| {
                    candidate.kind == StepKind::ToolResult
                        && candidate.tool_call_id.as_deref() == Some(call_id)
                })
                .map(|(result_position, _)| result_position)
        });
        let name = step
            .tool_name
            .clone()
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "tool_call".to_owned());
        let (end, status, open) = match closed_at {
            Some(result_position) => (result_position, steps[result_position].status, false),
            None => (position, step.status, true),
        };
        projected.push(TrajectorySpan {
            span_id: format!("span-{position}"),
            name,
            kind: StepKind::ToolCall,
            start: position,
            end,
            status,
            open,
        });
    }
    projected
}

/// The span covering the most steps; ties resolve to the earliest call.
#[must_use]
pub fn longest_span(trajectory: &Trajectory) -> Option<TrajectorySpan> {
    spans(trajectory)
        .into_iter()
        .max_by_key(|span| (span.end.saturating_sub(span.start), Reverse(span.start)))
}

/// Every span whose status is [`StepStatus::Error`], in step order.
#[must_use]
pub fn failed_spans(trajectory: &Trajectory) -> Vec<TrajectorySpan> {
    spans(trajectory)
        .into_iter()
        .filter(|span| span.status == StepStatus::Error)
        .collect()
}

/// The first span named `name`, if any.
///
/// Unnamed calls are named `tool_call`, so they are reachable by that name.
#[must_use]
pub fn span_by_name(trajectory: &Trajectory, name: &str) -> Option<TrajectorySpan> {
    spans(trajectory).into_iter().find(|span| span.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::model::TrajectoryStep;

    fn trajectory_with(steps: Vec<TrajectoryStep>) -> Trajectory {
        let mut trajectory = Trajectory::new("t-1").expect("valid id");
        trajectory.extend(steps);
        trajectory
    }

    #[test]
    fn pairs_calls_with_the_next_matching_result() {
        let trajectory = trajectory_with(vec![
            TrajectoryStep::new(StepKind::ToolCall, "{}")
                .with_tool_name("search")
                .with_tool_call_id("c1"),
            TrajectoryStep::new(StepKind::ToolResult, "hit")
                .with_tool_call_id("c1")
                .with_status(StepStatus::Error),
            TrajectoryStep::new(StepKind::ToolCall, "{}").with_tool_call_id("c2"),
        ]);
        let projected = spans(&trajectory);
        assert_eq!(projected.len(), 2);
        assert_eq!(projected[0].span_id, "span-0");
        assert_eq!((projected[0].start, projected[0].end), (0, 1));
        assert!(!projected[0].open);
        assert_eq!(projected[0].status, StepStatus::Error);
        assert_eq!(projected[1].name, "tool_call");
        assert!(projected[1].open);
        assert_eq!(projected[1].end, projected[1].start);
        assert_eq!(failed_spans(&trajectory).len(), 1);
    }

    #[test]
    fn results_before_the_call_do_not_pair() {
        let trajectory = trajectory_with(vec![
            TrajectoryStep::new(StepKind::ToolResult, "early").with_tool_call_id("c1"),
            TrajectoryStep::new(StepKind::ToolCall, "{}")
                .with_tool_name("calc")
                .with_tool_call_id("c1"),
        ]);
        let projected = spans(&trajectory);
        assert_eq!(projected.len(), 1);
        assert!(projected[0].open);
        assert_eq!(
            longest_span(&trajectory).map(|span| span.name),
            Some("calc".to_owned())
        );
        assert!(span_by_name(&trajectory, "calc").is_some());
        assert!(span_by_name(&trajectory, "missing").is_none());
    }
}
