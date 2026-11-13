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
//! - [`responses`] — `OpenAI` Responses API client (`POST /responses`), used by
//!   the `openai_account` provider and by `openai` with `api_mode: responses`.
//! - [`reasoning`] — per-family reasoning/thinking profiles, capabilities, and
//!   the wire encoders that turn a [`ReasoningConfig`] into request parameters.
//! - [`routing`] — `IntelliRouter` model groups, route compilation, and the
//!   failover [`RoutingModel`].
//! - [`oauth`] — `OpenAI` account device-authorization and token storage.
//! - [`dashscope`] — `DashScope` image, speech, and video generation over HTTP.
//!
//! The module root re-exports the types call sites need most: the configs, the
//! [`Model`] trait and its response types, both clients, the provider registry,
//! [`init_model`]/[`init_model_with`], and the reasoning/routing/oauth/media
//! entry points.

pub mod anthropic;
pub mod config;
pub mod dashscope;
pub mod factory;
pub mod model;
pub mod oauth;
pub mod openai;
pub mod provider;
pub mod reasoning;
pub mod registry;
pub mod responses;
pub mod routing;
pub mod sse;
pub(crate) mod transport;

pub use anthropic::AnthropicClient;
pub use config::{ApiMode, AuthMode, ModelClientConfig, ModelRequestConfig};
pub use dashscope::{
    AudioGenerationResponse, DashScopeConfig, DashScopeMediaClient, ImageGenerationResponse,
    ImageRequest, SpeechRequest, VideoGenerationResponse, VideoRequest,
};
pub use factory::{init_model, init_model_with};
pub use model::{
    AssistantTurn, ChatResponse, Choice, FinishReason, Model, ModelExt, StreamAccumulator,
    StreamChunk,
};
pub use oauth::{
    DeviceCode, FileTokenStore, InMemoryTokenStore, OAuthToken, OpenAiOAuthClient, TokenStore,
};
pub use openai::OpenAiCompatibleClient;
pub use provider::{
    normalize_provider_name, provider_profile, ProviderFlavor, ProviderProfile, ANTHROPIC_VERSION,
};
pub use reasoning::{
    apply_reasoning_plan, apply_reasoning_plan_with, find_reasoning_profile,
    resolve_reasoning_plan, resolve_reasoning_plan_with_endpoint, ReasoningCapability,
    ReasoningConfig, ReasoningMode, ReasoningPlan, ReasoningProfile, ReasoningProtocol,
};
pub use registry::{
    register_builtin_providers, register_model_provider, ModelProvider, ModelProviderRegistry,
};
pub use responses::{
    build_responses_body, parse_responses_response, parse_responses_usage, OpenAiResponsesClient,
    ResponsesStreamState,
};
pub use routing::{
    compile_model_selection, CompiledModelSelection, ModelSelection, ResolvedModel,
    ResolvedModelGroup, ResolvedRoute, ResolvedRoutingConfig, RoutingModel, RoutingStrategy,
};
pub use sse::{SseDecoder, SseEvent};
