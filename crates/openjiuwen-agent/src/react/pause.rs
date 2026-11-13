//! Persists and renders a paused `ReAct` turn.
//!
//! The loop calls [`pause_output`] whenever a turn cannot continue without the
//! caller: rail-recorded interrupts and client tool calls are persisted as an
//! [`InterruptionState`] and rendered as `__interaction__` frames.
use crate::interrupt::{InterruptRequest, InterruptionState};
use crate::output::{AgentChunk, AgentOutput, Observation, INTERACTION};
use openjiuwen_common::error::Result;
use openjiuwen_foundation::BaseMessage;
use openjiuwen_session::Session;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;

/// Persists and renders a paused turn.
///
/// The pending requests are stored under their tool call id (or interrupt id
/// when the request carries no call id), each becomes one `__interaction__`
/// frame delivered to the publisher and mirrored into the session, and the
/// returned output is a `result_type: "interrupt"` answer carrying those frames
/// and ids.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn pause_output(
    session: &Arc<dyn Session>,
    publisher: Option<&UnboundedSender<AgentChunk>>,
    iteration: usize,
    original_query: &str,
    assistant_message: Option<BaseMessage>,
    requests: &[InterruptRequest],
    answers: &BTreeMap<String, Value>,
    observations: Vec<Observation>,
) -> Result<AgentOutput> {
    let mut stored = BTreeMap::new();
    let mut frames = Vec::with_capacity(requests.len());
    let mut ids = Vec::with_capacity(requests.len());
    for (index, request) in requests.iter().enumerate() {
        let key = if request.tool_call_id.is_empty() {
            request.interrupt_id.clone()
        } else {
            request.tool_call_id.clone()
        };
        stored.insert(key, request.clone());
        let frame = request.to_frame(i64::try_from(index).unwrap_or(i64::MAX));
        ids.push(request.interrupt_id.clone());
        frames.push(frame);
    }
    let state = InterruptionState {
        iteration,
        original_query: original_query.to_owned(),
        assistant_message,
        requests: stored,
        answers: answers.clone(),
    };
    state.save(&**session).await?;

    for frame in &frames {
        let chunk = AgentChunk::new(INTERACTION, frame.index, frame.payload.clone());
        if let Some(publisher) = publisher {
            let _ = publisher.send(chunk.clone());
        }
        if let Err(error) = session.write_stream(chunk.to_stream_item()).await {
            tracing::warn!(%error, "failed to write an interaction frame to the session");
        }
    }

    Ok(AgentOutput::interrupt(frames, ids)
        .with_iterations_used(iteration)
        .with_observations(observations))
}
