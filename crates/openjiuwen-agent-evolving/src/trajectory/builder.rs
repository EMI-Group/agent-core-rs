//! Incremental construction of [`Trajectory`] values.
//!
//! Ports the accumulator role of Python
//! `openjiuwen.agent_evolving.trajectory.offline.builder` (and the message
//! projection in `trajectory/messages.py`) without the OTLP envelope: steps
//! are appended in call order and the trajectory assigns their indices.
//!
//! Messages are copied, never borrowed: [`Trajectory`] is `'static` and can be
//! stored, serialized, or sent to another task. Tool calls carried on a
//! [`BaseMessage`] are *not* expanded automatically — record them with
//! [`TrajectoryBuilder::push_tool_call`] so their call ids can pair with the
//! results recorded through [`TrajectoryBuilder::push_tool_result`].

use openjiuwen_common::error::Result;
use openjiuwen_core::foundation::prompt::{BaseMessage, MessageContent, Role};

use super::model::{StepKind, StepStatus, Trajectory, TrajectoryOutcome, TrajectoryStep};

/// Placeholder recorded for a non-text (multimodal) message payload.
const NON_TEXT_CONTENT_MARKER: &str = "[non-text content]";

/// Fluent accumulator for one [`Trajectory`].
///
/// Every `push_*` method consumes and returns the builder, so a recording
/// reads as a single chain:
///
/// ```
/// use openjiuwen_agent_evolving::trajectory::{StepKind, TrajectoryBuilder, TrajectoryOutcome};
///
/// let trajectory = TrajectoryBuilder::new("traj-1")?
///     .push_step(StepKind::Message, "hi")
///     .push_model_output("hello")
///     .finish(TrajectoryOutcome::Success);
/// assert_eq!(trajectory.len(), 2);
/// # Ok::<(), openjiuwen_common::error::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryBuilder {
    trajectory: Trajectory,
}

impl TrajectoryBuilder {
    /// Starts a builder for the trajectory `trajectory_id`.
    ///
    /// The id is validated by [`Trajectory::new`]; a blank id is an
    /// [`openjiuwen_common::error::Error::Validation`].
    pub fn new(trajectory_id: impl Into<String>) -> Result<Self> {
        Ok(Self {
            trajectory: Trajectory::new(trajectory_id)?,
        })
    }

    /// Sets the session identity.
    #[must_use]
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.trajectory = self.trajectory.with_session(session_id);
        self
    }

    /// Sets the evaluation case identity.
    #[must_use]
    pub fn with_case(mut self, case_id: impl Into<String>) -> Self {
        self.trajectory = self.trajectory.with_case(case_id);
        self
    }

    /// Adds one run metadata entry, replacing a previous value for the key.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.trajectory = self.trajectory.with_metadata(key, value);
        self
    }

    /// Appends one generic step of `kind` carrying `content`.
    #[must_use]
    pub fn push_step(mut self, kind: StepKind, content: impl Into<String>) -> Self {
        self.trajectory.push(TrajectoryStep::new(kind, content));
        self
    }

    /// Appends a [`StepKind::ToolCall`] step.
    ///
    /// `args_summary` is the (already summarized) argument text; the call id
    /// pairs the step with its result in [`super::spans`].
    #[must_use]
    pub fn push_tool_call(
        mut self,
        name: impl Into<String>,
        call_id: impl Into<String>,
        args_summary: impl Into<String>,
    ) -> Self {
        self.trajectory.push(
            TrajectoryStep::new(StepKind::ToolCall, args_summary)
                .with_tool_name(name)
                .with_tool_call_id(call_id),
        );
        self
    }

    /// Appends a [`StepKind::ToolResult`] step for the call `call_id`.
    #[must_use]
    pub fn push_tool_result(
        mut self,
        call_id: impl Into<String>,
        content: impl Into<String>,
        status: StepStatus,
    ) -> Self {
        self.trajectory.push(
            TrajectoryStep::new(StepKind::ToolResult, content)
                .with_tool_call_id(call_id)
                .with_status(status),
        );
        self
    }

    /// Appends one conversation message.
    ///
    /// The role becomes the step's role string; [`Role::System`] maps to
    /// [`StepKind::System`] and [`Role::Tool`] to [`StepKind::ToolResult`]
    /// (keeping the message's `tool_call_id`), everything else to
    /// [`StepKind::Message`]. Only text content is copied; a multimodal
    /// payload is recorded as `[non-text content]` instead of panicking or
    /// silently dropping the step.
    #[must_use]
    pub fn push_message(mut self, message: &BaseMessage) -> Self {
        let kind = match message.role {
            Role::System => StepKind::System,
            Role::Tool => StepKind::ToolResult,
            Role::User | Role::Assistant | Role::Other(_) => StepKind::Message,
        };
        let mut step = TrajectoryStep::new(kind, message_text(&message.content))
            .with_role(message.role.as_str());
        if let Some(call_id) = &message.tool_call_id {
            step = step.with_tool_call_id(call_id.clone());
        }
        self.trajectory.push(step);
        self
    }

    /// Appends a [`StepKind::Model`] step for raw model output.
    ///
    /// The step's role is `assistant`, matching the wire role of a completion.
    #[must_use]
    pub fn push_model_output(mut self, content: impl Into<String>) -> Self {
        self.trajectory
            .push(TrajectoryStep::new(StepKind::Model, content).with_role("assistant"));
        self
    }

    /// Number of steps recorded so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.trajectory.len()
    }

    /// Whether no step has been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.trajectory.is_empty()
    }

    /// The trajectory recorded so far, for inspection before finishing.
    #[must_use]
    pub fn trajectory(&self) -> &Trajectory {
        &self.trajectory
    }

    /// Finishes the recording with `outcome`.
    #[must_use]
    pub fn finish(mut self, outcome: TrajectoryOutcome) -> Trajectory {
        self.trajectory.outcome = outcome;
        self.trajectory
    }

    /// Finishes the recording, keeping the current outcome
    /// ([`TrajectoryOutcome::Unknown`] unless it was set elsewhere).
    #[must_use]
    pub fn build(self) -> Trajectory {
        self.trajectory
    }
}

/// Builds a trajectory from an already-recorded message list.
///
/// Steps get contiguous indices starting at `0`, in message order. The result
/// is the same as chaining [`TrajectoryBuilder::push_message`] over every
/// message; `session_id` is applied when supplied.
pub fn from_messages(
    trajectory_id: impl Into<String>,
    session_id: Option<&str>,
    messages: &[BaseMessage],
) -> Result<Trajectory> {
    let mut builder = TrajectoryBuilder::new(trajectory_id)?;
    if let Some(session_id) = session_id {
        builder = builder.with_session(session_id);
    }
    for message in messages {
        builder = builder.push_message(message);
    }
    Ok(builder.build())
}

/// Copies a message payload as text, marking multimodal content.
fn message_text(content: &MessageContent) -> String {
    match content.as_text() {
        Some(text) => text.to_owned(),
        None => NON_TEXT_CONTENT_MARKER.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use openjiuwen_core::foundation::prompt::BaseMessage;

    use super::*;

    #[test]
    fn pushes_record_roles_and_call_ids() {
        let message = BaseMessage::tool("call-1", "42");
        let trajectory = TrajectoryBuilder::new("t-1")
            .expect("valid id")
            .push_message(&message)
            .push_tool_call("calc", "call-2", "{}")
            .push_tool_result("call-2", "7", StepStatus::Ok)
            .finish(TrajectoryOutcome::Success);
        assert_eq!(trajectory.steps()[0].kind, StepKind::ToolResult);
        assert_eq!(
            trajectory.steps()[0].tool_call_id.as_deref(),
            Some("call-1")
        );
        assert_eq!(trajectory.steps()[1].tool_name.as_deref(), Some("calc"));
        assert!(trajectory.is_successful());
    }

    #[test]
    fn non_text_content_is_marked_not_dropped() {
        use openjiuwen_core::foundation::prompt::{ContentPart, MessageContent};

        let message = BaseMessage::user(MessageContent::Parts(vec![ContentPart::ImageUrl {
            image_url: openjiuwen_core::foundation::prompt::ImageUrl {
                url: "https://example.invalid/x.png".to_owned(),
                detail: None,
            },
        }]));
        let trajectory = from_messages("t-1", None, &[message]).expect("valid id");
        assert_eq!(trajectory.steps()[0].content, "[non-text content]");
        assert_eq!(trajectory.steps()[0].role.as_deref(), Some("user"));
    }

    #[test]
    fn from_messages_assigns_contiguous_indices() {
        let messages = [BaseMessage::system("s"), BaseMessage::user("u")];
        let trajectory = from_messages("t-1", Some("session-1"), &messages).expect("valid id");
        let indices: Vec<usize> = trajectory.steps().iter().map(|step| step.index).collect();
        assert_eq!(indices, vec![0, 1]);
        assert_eq!(trajectory.session_id.as_deref(), Some("session-1"));
        assert_eq!(trajectory.steps()[0].kind, StepKind::System);
    }
}
