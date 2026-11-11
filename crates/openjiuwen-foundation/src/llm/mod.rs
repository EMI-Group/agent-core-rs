//! LLM clients, the [`model::Model`] trait, and the provider factory.
//!
//! Ports `openjiuwen/core/foundation/llm/`. [`model`] owns the provider-neutral
//! surface, [`config`] the serializable configs, [`provider`] the provider table,
//! [`registry`] the pluggable factory, [`factory`] the `init_model` entry point,
//! and [`sse`] the incremental Server-Sent-Events decoder shared by both clients.
//!
//! - [`openai`] — OpenAI-compatible chat-completions client (OpenAI, Pangu,
//!   Qwen/DashScope, DeepSeek, Moonshot, OpenRouter, Ollama, LM Studio, ...).
//! - [`anthropic`] — Anthropic Messages client.

pub mod anthropic;
pub mod config;
pub mod factory;
pub mod model;
pub mod openai;
pub mod provider;
pub mod registry;
pub mod sse;
