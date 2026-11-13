//! Per-request and per-context options for the context engine.
//!
//! These are the value types [`ContextEngine`] accepts: [`PrepareOptions`] for
//! one payload assembly and [`ContextOptions`] for one context creation. They
//! live beside [`engine`](crate::engine) and are re-exported there, so
//! `openjiuwen_context::engine::ContextOptions` keeps working.
//!
//! [`ContextEngine`]: crate::engine::ContextEngine
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{BaseMessage, TokenCounter};
use openjiuwen_session::Session;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::processor::ProcessorConfig;
use crate::recall::RecallArchiveStore;
use crate::session_memory::SessionMemoryStore;

/// Session id used when a context is created without a session.
pub const DEFAULT_SESSION_ID: &str = "default_session_id";

/// Per-request overrides for [`ContextEngine::prepare`].
///
/// [`ContextEngine::prepare`]: crate::engine::ContextEngine::prepare
#[derive(Debug, Clone)]
pub struct PrepareOptions {
    /// System messages, placed before the selected context messages.
    pub system_messages: Vec<BaseMessage>,
    /// Maximum messages to send, system messages included.
    pub window_size: Option<usize>,
    /// Maximum dialogue rounds to send.
    pub dialogue_rounds: Option<usize>,
    /// Tool schemas passed through to the provider untouched.
    pub tools: Vec<Value>,
    /// Token the caller cancels to abandon preparation.
    pub cancellation: CancellationToken,
    /// Deadline for the whole preparation, when one is required.
    pub timeout: Option<Duration>,
}

impl Default for PrepareOptions {
    fn default() -> Self {
        Self {
            system_messages: Vec::new(),
            window_size: None,
            dialogue_rounds: None,
            tools: Vec::new(),
            cancellation: CancellationToken::new(),
            timeout: None,
        }
    }
}

/// Everything [`ContextEngine::create_context_with`] accepts besides the id.
///
/// [`ContextEngine::create_context_with`]: crate::engine::ContextEngine::create_context_with
#[derive(Default, Clone)]
pub struct ContextOptions {
    /// Session the context belongs to; `None` uses [`DEFAULT_SESSION_ID`].
    pub session: Option<Arc<dyn Session>>,
    /// Initial message history.
    pub history: Vec<BaseMessage>,
    /// Pipeline override; `None` uses [`ContextConfig::processors`].
    ///
    /// [`ContextConfig::processors`]: crate::config::ContextConfig::processors
    pub processors: Option<Vec<ProcessorConfig>>,
    /// Counter override; `None` builds the configured one.
    pub token_counter: Option<Arc<dyn TokenCounter + Send + Sync>>,
    /// Model handed to summarisation processors.
    pub model: Option<Arc<dyn Model>>,
    /// Recall archive store override; `None` uses the engine's store, then the
    /// configuration default.
    pub recall: Option<Arc<dyn RecallArchiveStore>>,
    /// Session-memory store override; `None` uses the engine's store, then the
    /// configuration default.
    pub session_memory: Option<Arc<dyn SessionMemoryStore>>,
}

impl std::fmt::Debug for ContextOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextOptions")
            .field(
                "session_id",
                &self.session.as_ref().map(|session| session.id()),
            )
            .field("history", &self.history.len())
            .field("processors", &self.processors)
            .field("has_token_counter", &self.token_counter.is_some())
            .field("has_model", &self.model.is_some())
            .field("has_recall", &self.recall.is_some())
            .field("has_session_memory", &self.session_memory.is_some())
            .finish_non_exhaustive()
    }
}
