//! Foundation layer of the openJiuwen kernel: messages and prompts, LLM clients,
//! tools, and storage backends.
//!
//! This crate ports the Python package `openjiuwen.core.foundation`
//! (`prompt/`, `llm/`, `tool/`, `store/`) and is the lowest provider seam of the
//! SDK: every other crate invokes models and tools only through the traits
//! defined here.
//!
//! - [`prompt`] — [`prompt::Role`], [`prompt::MessageContent`], [`prompt::ContentPart`],
//!   [`prompt::BaseMessage`], [`prompt::ToolCall`], [`prompt::Usage`],
//!   [`prompt::PromptTemplate`], and token helpers.
//! - [`llm`] — the [`llm::Model`] trait, [`llm::ModelClientConfig`],
//!   [`llm::ModelRequestConfig`], [`llm::ChatResponse`], [`llm::StreamChunk`],
//!   the OpenAI-compatible and Anthropic clients, [`llm::init_model`], and the
//!   provider registry.
//! - [`tool`] — [`tool::ToolCard`], the [`tool::Tool`] trait, [`tool::ToolContext`],
//!   [`tool::FunctionTool`], [`tool::RestfulApiTool`], and MCP support behind the
//!   `mcp` feature.
//! - [`store`] — the [`store::KvStore`], [`store::DocumentStore`],
//!   [`store::VectorStore`], [`store::GraphStore`], and [`store::ObjectStore`]
//!   traits with in-memory, local-filesystem, and optional SQLite backends.
//!
//! All I/O is async on tokio, public data types are `serde`-serializable with
//! snake_case field names, errors use [`openjiuwen_common::error::Error`], and no
//! secret is ever logged.
#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::too_many_arguments,
    clippy::struct_excessive_bools,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::items_after_statements,
    clippy::unused_async,
    clippy::significant_drop_tightening,
    clippy::manual_let_else,
    clippy::needless_pass_by_value
)]

pub mod llm;
pub mod prompt;
pub mod store;
pub mod tool;

/// The crate name, useful in diagnostics and documentation.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
/// The crate version.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
