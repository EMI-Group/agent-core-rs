//! The `ReAct` model/tool loop and its configuration.
//!
//! This round declares the configuration module; the agent, the iteration loop,
//! the model step, the resume replay, and the prompt assembly follow in later
//! rounds.

/// Configuration of the `ReAct` agent and the max-iteration policy.
pub mod config;
