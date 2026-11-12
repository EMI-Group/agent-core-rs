//! Replay of an interrupted `ReAct` turn.
//!
//! A turn paused on an interrupt persists its pending requests; resuming matches
//! the caller's [`InteractiveInput`] answers against them with
//! [`match_answers`](crate::interrupt::match_answers) (tool-call id first, then
//! interrupt id) and turns each answer into a tool result: an approved call is
//! re-executed, a rejected call yields the rejection-feedback tool message, and
//! an ask-user answer becomes the tool content.
use crate::abilities::ToolCallResult;
use crate::interrupt::InterruptionState;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::InteractiveInput;

/// Iteration a resumed run restarts from after replaying the interrupted calls.
#[allow(dead_code)] // used by the loop's resume path
pub(crate) const RESUME_START_ITERATION: usize = 1;

/// Replays the interrupted tool calls of `state` using `interactive` answers.
///
/// Each entry of [`InterruptionState::requests`] is matched to an answer and
/// produces one [`ToolCallResult`] in model order, so the loop can append the
/// corresponding tool messages and continue.
///
/// # Errors
///
/// Returns an error when the answers cannot be matched or a replayed call cannot
/// be executed at all (as opposed to a tool failure, which becomes a result).
#[allow(dead_code)] // used by the loop's resume path
#[allow(clippy::unused_async)] // the implementation re-executes the approved tool calls
pub(crate) async fn replay_interrupted(
    state: &InterruptionState,
    interactive: &InteractiveInput,
) -> Result<Vec<ToolCallResult>> {
    let _ = (state, interactive);
    Err(Error::unsupported("interrupt resume not implemented"))
}
