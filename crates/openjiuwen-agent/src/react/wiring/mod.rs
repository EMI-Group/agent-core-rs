//! Behavior tests for the `ReAct` loop's side seams.
//!
//! The [`callbacks`] suite drives a real `ReActAgent` end to end and records
//! every lifecycle event it fires; the [`steering`] suite proves where a queued
//! message lands in the model window. Both suites share an in-process echo tool
//! and a `ScriptedModel`, so they stay offline and deterministic.
//!
//! The module tree only compiles under `cfg(test)` (the parent `react` module
//! gates `mod wiring`), so it runs in the crate's lib test binary.
mod callbacks;
mod steering;

use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::{Model, Tool, ToolCard, ToolContext};
use serde_json::{json, Value};

use crate::card::AgentCard;
use crate::react::ReActAgent;

/// A tool that echoes its parsed inputs back as the result text.
struct EchoTool {
    card: ToolCard,
}

#[async_trait::async_trait]
impl Tool for EchoTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        Ok(Value::String(format!("echo: {inputs}")))
    }
}

/// Builds the `echo` tool the wiring suites mount.
fn echo_tool() -> Arc<dyn Tool> {
    Arc::new(EchoTool {
        card: ToolCard::new("echo", "Echoes its input", json!({"type": "object"})).with_id("echo"),
    })
}

/// Builds an agent named and identified as `react-test`, driven by `model`.
fn agent(model: Arc<dyn Model>) -> ReActAgent {
    let agent = ReActAgent::new(AgentCard::new("react-test").with_id("react-test"));
    agent.set_model(model);
    agent
}
