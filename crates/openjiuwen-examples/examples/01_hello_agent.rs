//! `01_hello_agent` — the smallest complete agent: a model, an agent, one `invoke`.
//!
//! It teaches the three seams every later example builds on: the `AgentCard`
//! that identifies the agent, the `ReActAgentConfig` that names the model, and
//! the `Agent` trait's `invoke`, which turns one JSON input into an
//! `AgentOutput`. The streaming surface is shown last: the same question through
//! `Agent::stream`, which yields typed frames instead of one answer.
//!
//! Offline (the default) the model is a `ScriptedModel`, so the run is
//! deterministic and needs no network. With `OPENJIUWEN_LIVE=1`, a model name and
//! a credential, the very same code talks to a real provider instead.
//!
//! Python counterpart: `examples/react_agent/build_react_agent.ipynb` (the SDK
//! has no standalone hello-world script), reduced to the first agent turn.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements
)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use futures::StreamExt;
use openjiuwen::core::foundation::Model;
use openjiuwen::core::single_agent::{Agent, AgentCard, ReActAgent, ReActAgentConfig, ResultType};
use serde_json::json;

/// Model name written into the agent config (the live model comes from the env).
const MODEL_NAME: &str = "gpt-4o-mini";

/// Provider written into the agent config.
const MODEL_PROVIDER: &str = "openai";

/// The single question this example asks.
const QUESTION: &str = "What is the capital of France?";

/// The answer the offline scripted model returns.
const SCRIPTED_ANSWER: &str = "Paris is the capital of France.";

/// Upper bound on one model turn, so a misconfigured provider cannot hang the demo.
const TURN_TIMEOUT: Duration = Duration::from_secs(60);

/// Builds the agent both branches use: a card, a config, and a model.
fn build_agent(model: Arc<dyn Model>) -> ReActAgent {
    let agent = ReActAgent::new(AgentCard::new("hello-agent"));

    let mut config = ReActAgentConfig::new();
    config
        .configure_model(MODEL_NAME, MODEL_PROVIDER)
        .configure_max_iterations(4);

    agent.configure(config);
    agent.set_model(model);
    agent
}

#[tokio::main]
async fn main() -> Result<()> {
    let mode = common::setup()?;
    common::banner::title("01 hello agent");

    // Each scripted model serves one turn, and the example makes two calls
    // (`invoke`, then `stream`), so the offline branch builds one model per call.
    // A live client is reusable, so it is cloned instead.
    let (agent, stream_agent) = if mode.is_live() {
        let provider = mode.provider().unwrap_or(MODEL_PROVIDER);
        let model_name = mode.model().unwrap_or(MODEL_NAME);
        common::banner::note(&format!("live provider: {provider}/{model_name}"));
        let model = common::live_model().await?;
        (build_agent(Arc::clone(&model)), build_agent(model))
    } else {
        common::banner::note(
            "using a scripted model; set OPENJIUWEN_LIVE=1 plus a model name and credential \
             to talk to a real provider",
        );
        (
            build_agent(common::scripted(SCRIPTED_ANSWER)),
            build_agent(common::scripted(SCRIPTED_ANSWER)),
        )
    };

    common::banner::step(&format!(
        "card: {} (id {})",
        agent.card().name,
        agent.card().id
    ));
    common::banner::step(&format!("invoking with: {QUESTION}"));

    let output = tokio::time::timeout(TURN_TIMEOUT, agent.invoke(json!(QUESTION), None))
        .await
        .context("the invoke timed out")??;

    let result_type = match output.result_type {
        ResultType::Answer => "answer",
        ResultType::Error => "error",
        ResultType::Interrupt => "interrupt",
    };
    common::banner::note(&format!("result type: {result_type}"));
    common::banner::note(&format!(
        "iterations used: {}",
        output.iterations_used.unwrap_or_default()
    ));
    common::banner::answer(&output.output);

    common::banner::step("streaming the same question through a second agent");

    let frames = tokio::time::timeout(
        TURN_TIMEOUT,
        stream_agent
            .stream(json!(QUESTION), None)
            .collect::<Vec<_>>(),
    )
    .await
    .context("the stream timed out")?;

    let mut frame_count = 0_usize;
    let mut streamed_answer = String::new();
    for frame in frames {
        let frame = frame?;
        frame_count += 1;
        tracing::debug!(kind = %frame.kind, index = frame.index, "stream frame");
        // The terminal answer frame carries the same shape as `AgentOutput`.
        if frame.kind == "answer" {
            if let Some(text) = frame
                .payload
                .get("output")
                .and_then(serde_json::Value::as_str)
            {
                streamed_answer.push_str(text);
            }
        }
    }

    common::banner::note(&format!("stream produced {frame_count} frames"));
    common::banner::answer(&streamed_answer);

    Ok(())
}
