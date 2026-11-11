//! Foundation layer of the openJiuwen kernel: messages and prompts, LLM clients,
//! tools, and storage backends.
//!
//! This crate ports the Python package `openjiuwen.core.foundation`
//! (`prompt/`, `llm/`, `tool/`, `store/`) and is the lowest provider seam of the
//! SDK: every other crate invokes models and tools only through the traits
//! defined here.
//!
//! - [`prompt`] — [`Role`], [`MessageContent`], [`ContentPart`], [`BaseMessage`],
//!   [`ToolCall`], [`Usage`], [`PromptTemplate`], and token helpers.
//! - [`llm`] — the [`Model`] trait, [`ModelClientConfig`], [`ModelRequestConfig`],
//!   [`ChatResponse`], [`StreamChunk`], the OpenAI-compatible and Anthropic
//!   clients, [`init_model`], and the provider registry.
//! - [`tool`] — [`ToolCard`], the [`Tool`] trait, [`ToolContext`],
//!   [`FunctionTool`], [`RestfulApiTool`], and MCP support behind the `mcp` feature.
//! - [`store`] — the [`KvStore`], [`DocumentStore`], [`VectorStore`],
//!   [`GraphStore`], and [`ObjectStore`] traits with in-memory,
//!   local-filesystem, and optional SQLite backends.
//!
//! All I/O is async on tokio, public data types are `serde`-serializable with
//! `snake_case` field names, errors use [`openjiuwen_common::error::Error`], and
//! no secret is ever logged.
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

// --- messages and prompts ---------------------------------------------------

pub use prompt::{
    count_message_tokens, truncate_messages, ApproxTokenCounter, AudioPart, BaseMessage,
    ContentPart, FilePart, ImageUrl, MessageContent, PromptTemplate, Role, TemplateContent,
    TokenCounter, ToolCall, ToolCallAccumulator, ToolCallDelta, Usage,
};

// --- models -----------------------------------------------------------------

pub use llm::{
    init_model, init_model_with, normalize_provider_name, provider_profile,
    register_builtin_providers, register_model_provider, AnthropicClient, AssistantTurn,
    ChatResponse, Choice, FinishReason, Model, ModelClientConfig, ModelExt, ModelProvider,
    ModelProviderRegistry, ModelRequestConfig, OpenAiCompatibleClient, ProviderFlavor,
    ProviderProfile, StreamAccumulator, StreamChunk,
};

// --- tools ------------------------------------------------------------------

pub use tool::{
    apply_defaults, ensure_object_schema, openai_tools, render_payload_text, render_tool_output,
    schema_for, validate_inputs, ApiParamMapper, FunctionTool, FunctionToolBuilder, HttpMethod,
    ParamLocation, ParamMap, RestfulApiConfig, RestfulApiTool, Tool, ToolCard, ToolContext,
    ToolExecution, ToolExposure, ToolHandler, ToolInfo, ToolOutput, EMPTY_FAILURE_TEXT,
    EMPTY_SUCCESS_TEXT,
};

#[cfg(feature = "mcp")]
pub use tool::mcp::{
    connect_mcp_server, extract_content, load_mcp_tools, McpClient, McpClientType, McpRemoteTool,
    McpServerConfig, McpTool, McpToolOptions, McpToolResult, RmcpClient,
};

// --- storage ----------------------------------------------------------------

pub use store::{
    matches as filter_matches, matches_all as filter_matches_all, CollectionSchema, Direction,
    Document, DocumentStore, FieldSchema, GraphEdge, GraphNode, GraphQuery, GraphStore,
    InMemoryDocumentStore, InMemoryGraphStore, InMemoryKvStore, InMemoryObjectStore,
    InMemoryVectorStore, KvPipeline, KvStore, KvValue, LocalFileDocumentStore, LocalFileGraphStore,
    LocalFileKvStore, LocalFileVectorStore, LocalObjectStore, ObjectMeta, ObjectStore, Query,
    SortOrder, VectorDataType, VectorQuery, VectorSearchResult, VectorStore,
};

#[cfg(feature = "sqlite")]
pub use store::{SqliteDocumentStore, SqliteKvStore};
