//! LLM clients, the [`model::Model`] trait, and the provider factory.
//!
//! Ports `openjiuwen/core/foundation/llm/`. [`model`] owns the provider-neutral
//! surface, [`config`] the serializable configs, [`provider`] the provider table,
//! [`registry`] the pluggable factory, [`factory`] the `init_model` entry point,
//! and [`sse`] the incremental `Server-Sent Events` decoder shared by both clients.
//!
//! - [`openai`] — OpenAI-compatible chat-completions client (`OpenAI`, `Pangu`,
//!   `Qwen`/`DashScope`, `DeepSeek`, `Moonshot`, `OpenRouter`, `Ollama`,
//!   `LM Studio`, ...).
//! - [`anthropic`] — Anthropic Messages client.
//!
//! The module root re-exports the types call sites need most: the configs, the
//! [`Model`] trait and its response types, both clients, the provider registry,
//! and [`init_model`]/[`init_model_with`].

pub mod anthropic;
pub mod config;
pub mod factory;
pub mod model;
pub mod openai;
pub mod provider;
pub mod registry;
pub mod sse;
pub(crate) mod transport;

pub use anthropic::AnthropicClient;
pub use config::{ModelClientConfig, ModelRequestConfig};
pub use factory::{init_model, init_model_with};
pub use model::{
    AssistantTurn, ChatResponse, Choice, FinishReason, Model, ModelExt, StreamAccumulator,
    StreamChunk,
};
pub use openai::OpenAiCompatibleClient;
pub use provider::{
    normalize_provider_name, provider_profile, ProviderFlavor, ProviderProfile, ANTHROPIC_VERSION,
};
pub use registry::{
    register_builtin_providers, register_model_provider, ModelProvider, ModelProviderRegistry,
};
pub use sse::{SseDecoder, SseEvent};
