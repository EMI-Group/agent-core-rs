//! The `ReAct` model/tool loop and its configuration.
//!
//! This round declares the configuration and model-step modules; the agent, the
//! iteration loop, the resume replay, and the prompt assembly follow in later
//! rounds.

/// Configuration of the `ReAct` agent and the max-iteration policy.
pub mod config;

/// A single model call and the normalized [`ModelReply`](model_step::ModelReply).
pub mod model_step;
