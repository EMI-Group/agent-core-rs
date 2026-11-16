//! Chat-agent adapters that bind a tuning optimizer to an agent-like runner.
//!
//! Ports `openjiuwen/dev_tools/tune/chat_agent/` (`chat_config.py`,
//! `chat_agent.py`) without Python's runtime: a [`ChatConfig`] carries the model
//! and prompt settings a chat agent is built from, [`ChatAgent`] assembles the
//! conversation from one candidate prompt plus one case and implements the
//! tuning [`PromptRunner`] seam, and the agent's own adapters
//! ([`ChatAgent::evaluator`], [`ChatAgent::optimize`], [`ChatAgent::train`]) bind
//! the optimizers and the trainer to that seam.
//!
//! The execution step is caller-supplied: [`ModelChatRunner`] answers through an
//! `openjiuwen_foundation::llm::Model`, [`ScriptedChatRunner`] replays canned
//! replies for offline tests, and a host runtime can implement [`ChatRunner`]
//! itself. Nothing here reads the environment, opens a network connection, or
//! consults a clock, so a tuning run is fully deterministic.
//!
//! ```
//! # use std::sync::Arc;
//! # use openjiuwen_dev_tools::tune::{
//! #     Case, ChatAgent, ChatConfig, ScriptedChatRunner,
//! # };
//! # use std::collections::BTreeMap;
//! # use serde_json::Value;
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<(), openjiuwen_common::error::Error> {
//! let config = ChatConfig::new("agent-1", "gpt-4o-mini").with_user_prompt("{{question}}");
//! let runner = Arc::new(ScriptedChatRunner::new().with_reply("2 + 2", "4"));
//! let agent = ChatAgent::with_runner(config, runner)?;
//!
//! let mut inputs = BTreeMap::new();
//! inputs.insert("question".to_owned(), Value::String("2 + 2?".to_owned()));
//! let mut label = BTreeMap::new();
//! label.insert("answer".to_owned(), Value::String("4".to_owned()));
//! let case = Case::new(inputs, label)?;
//!
//! let result = agent.optimize("Answer the question.", &[case]).await?;
//! assert!((result.score - 1.0).abs() < 1e-9);
//! # Ok(())
//! # }
//! ```
//!
//! [`PromptRunner`]: crate::tune::evaluator::PromptRunner

mod agent;
mod config;
mod runner;

pub use self::agent::{ChatAgent, CANDIDATE_BLOCK_HEADING};
pub use self::config::{create_chat_config, ChatConfig};
pub use self::runner::{ChatCall, ChatRunner, ModelChatRunner, ScriptedChatRunner};
