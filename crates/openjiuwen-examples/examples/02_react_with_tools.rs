//! `02_react_with_tools` — two tools on one agent, and one complete tool round.
//!
//! A `ReAct` agent exists to call tools. This example mounts the two ways of
//! defining one:
//!
//! - `word_count` is written with the `#[tool]` attribute macro: an `async fn`
//!   over a `serde`/`schemars` argument struct, from which the macro derives the
//!   JSON Schema, the `Tool` implementation and the `Arc<dyn Tool>` constructor.
//! - `shout` is a hand-written `FunctionTool` built from a JSON schema and a
//!   closure, for tools whose schema is dynamic or hand-tuned.
//!
//! The scripted model asks for `word_count` on its first turn and answers on its
//! second, so the printed observations show the tool round a real model would
//! drive: prompt → tool call → tool result → final answer.
//!
//! Offline only: a live provider would need its own prompt and would not
//! deterministically choose `word_count`, which is the point of this demo.
//!
//! Python counterpart: `examples/rl_calculator/tools.py` (`@tool`-decorated
//! functions) plus the first tool turn of `examples/react_agent`.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements,
    // `#[tool]` requires an `async fn`; this tool answers without awaiting.
    clippy::unused_async
)]

mod common;

use std::time::Duration;

use anyhow::{ensure, Context, Result};
use openjiuwen::core::foundation::{FunctionTool, ToolContext};
use openjiuwen::core::single_agent::{Agent, AgentCard, ReActAgent, ReActAgentConfig, ResultType};
use openjiuwen_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

/// Model name written into the agent config; the scripted model ignores it.
const MODEL_NAME: &str = "gpt-4o-mini";

/// Provider written into the agent config.
const MODEL_PROVIDER: &str = "openai";

/// The sentence handed to the `word_count` tool by the scripted model.
const SENTENCE: &str = "the quick brown fox jumps over the lazy dog";

/// Upper bound on the whole agent run.
const RUN_TIMEOUT: Duration = Duration::from_secs(60);

/// Arguments of the macro-defined tool, deserialized and schema-derived by `#[tool]`.
#[derive(Debug, Deserialize, JsonSchema)]
struct WordCountArgs {
    /// The text whose whitespace-separated words are counted.
    text: String,
}

/// Count the whitespace-separated words in a string.
#[tool(
    name = "word_count",
    description = "Count the whitespace-separated words in a string.",
    crate = "::openjiuwen::core::foundation",
    common_crate = "::openjiuwen::core::common"
)]
async fn word_count(args: WordCountArgs) -> openjiuwen::core::common::error::Result<Value> {
    Ok(json!({ "words": args.text.split_whitespace().count() }))
}

/// The hand-built tool: upper-cases its input with a plain closure handler.
fn shout_tool() -> Result<std::sync::Arc<dyn openjiuwen::core::foundation::Tool>> {
    let tool = FunctionTool::builder()
        .name("shout")
        .description("Return the given text in upper case.")
        .input_schema(json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "description": "Text to upper-case." }
            },
            "required": ["text"]
        }))
        .handler(|args: Value, _ctx: ToolContext| async move {
            let text = args.get("text").and_then(Value::as_str).unwrap_or_default();
            Ok(json!({ "shouted": text.to_uppercase() }))
        })
        .build()?;
    Ok(tool.boxed())
}

#[tokio::main]
async fn main() -> Result<()> {
    let _mode = common::setup()?;
    common::banner::title("02 react with tools");

    let agent = ReActAgent::new(AgentCard::new("tools-agent"));
    let mut config = ReActAgentConfig::new();
    config
        .configure_model(MODEL_NAME, MODEL_PROVIDER)
        .configure_max_iterations(4);
    agent.configure(config);

    // Turn one: ask for `word_count`; turn two: answer once the result is fed back.
    agent.set_model(common::scripted_tool_round(
        "word_count",
        json!({ "text": SENTENCE }),
        format!(
            "The sentence has {} words.",
            SENTENCE.split_whitespace().count()
        ),
    ));

    common::banner::step("mounting two tools");
    for mounted in [word_count_tool(), shout_tool()?] {
        let outcome = agent.add_tool(mounted);
        common::banner::note(&format!("mounted {} ({})", outcome.name, outcome.reason));
    }

    common::banner::step("tools the agent will advertise to the model");
    for card in agent.tools() {
        common::banner::note(&format!("{} — {}", card.name, card.description));
    }

    common::banner::step(&format!("asking about: \"{SENTENCE}\""));
    let output = tokio::time::timeout(
        RUN_TIMEOUT,
        agent.invoke(
            json!(format!("How many words are in \"{SENTENCE}\"?")),
            None,
        ),
    )
    .await
    .context("the agent run timed out")??;

    ensure!(
        output.result_type == ResultType::Answer,
        "expected an answer after the tool round, got {:?}",
        output.result_type
    );
    ensure!(
        !output.observations.is_empty(),
        "expected at least one tool observation"
    );

    for observation in &output.observations {
        common::banner::note(&format!(
            "observation from {}: {}",
            observation.name.as_deref().unwrap_or("unknown tool"),
            observation.content
        ));
    }
    common::banner::note(&format!(
        "iterations used: {}",
        output.iterations_used.unwrap_or_default()
    ));
    common::banner::answer(&output.output);

    Ok(())
}
