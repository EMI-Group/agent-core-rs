//! The [`Agent`] seam every runnable agent implements.
use crate::card::AgentCard;
use crate::output::{AgentChunk, AgentOutput};
use futures::stream::BoxStream;
use openjiuwen_common::error::Result;
use openjiuwen_session::Session;
use serde_json::Value;
use std::sync::Arc;
/// A runnable agent.
///
/// This is the seam consumed by `openjiuwen-runner` and by composite agents:
/// the runner injects the model and tools, then drives [`Agent::invoke`] or
/// [`Agent::stream`]. Implementations are shared as `Arc<dyn Agent>`.
#[async_trait::async_trait]
pub trait Agent: Send + Sync {
    /// Returns the agent's card.
    fn card(&self) -> &AgentCard;
    /// Runs one invocation to completion.
    ///
    /// The returned [`AgentOutput`] has a `result_type` of `answer`, `error`,
    /// or `interrupt`; an interrupted run carries state frames and interrupt
    /// ids that a later invocation can resume from.
    ///
    /// # Errors
    ///
    /// Returns an error when the invocation cannot run at all (invalid input,
    /// unconfigured model, cancelled session); recoverable model and tool
    /// failures are reported inside the output instead.
    async fn invoke(&self, inputs: Value, session: Option<Arc<dyn Session>>)
        -> Result<AgentOutput>;
    /// Runs one invocation, yielding [`AgentChunk`] frames as they are produced.
    fn stream(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>>;
}
