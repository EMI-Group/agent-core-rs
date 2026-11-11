//! Messages, content parts, tool calls, prompt templates, and token helpers.
//!
//! Ports `openjiuwen/core/foundation/prompt/` plus the message schema of
//! `openjiuwen/core/foundation/llm/schema/`. Every other module in the crate
//! (and every crate above it) builds prompts from these types, so they stay
//! provider-neutral: wire-specific shapes are produced by the provider clients
//! in [`crate::llm`].
//!
//! - [`role::Role`] — `system`/`user`/`assistant`/`tool` plus a tolerant fallback.
//! - [`content::MessageContent`] — text or a list of [`content::ContentPart`], both accepted from JSON.
//! - [`message::BaseMessage`] — one struct for all four roles, with constructors.
//! - [`tool_call`] — `ToolCall`, stream deltas, and the delta accumulator.
//! - [`usage::Usage`] — token accounting shared by the clients.
//! - [`template::PromptTemplate`] — `{{slot}}` rendering and message-list conversion.
//! - [`tokens`] — token counting and message truncation.

mod assembler;
pub mod content;
pub mod message;
pub mod role;
pub mod template;
pub mod tokens;
pub mod tool_call;
pub mod usage;

pub use content::{AudioPart, ContentPart, FilePart, ImageUrl, MessageContent};
pub use message::BaseMessage;
pub use role::Role;
pub use template::{PromptTemplate, TemplateContent};
#[cfg(feature = "tiktoken")]
pub use tokens::TiktokenCounter;
pub use tokens::{count_message_tokens, truncate_messages, ApproxTokenCounter, TokenCounter};
pub use tool_call::{ToolCall, ToolCallAccumulator, ToolCallDelta};
pub use usage::Usage;
