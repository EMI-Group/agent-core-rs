//! A [`CompletionModel`] adapter over a foundation `Model`.
//!
//! The rest of this module talks to one-prompt-in/one-text-out models; production
//! syphony deployments hold an `Arc<dyn openjiuwen_foundation::llm::Model>` (an
//! `OpenAiCompatibleClient`, an `AnthropicClient`, a routing model, …).
//! [`FoundationCompletionModel`] bridges the two: it renders the prompt as a user
//! message (optionally prefixed by a system message), forwards the temperature and
//! token cap through a [`ModelRequestConfig`](openjiuwen_foundation::llm::ModelRequestConfig), calls
//! [`Model::invoke`](openjiuwen_foundation::llm::Model::invoke), and returns the first
//! choice's plain text.
//!
//! The adapter performs no retries of its own — the foundation client owns its retry
//! policy — and it never streams: every stage in this crate wants one complete reply.
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_foundation::{BaseMessage, Model, ModelRequestConfig};

use crate::error::{Error, Result};

use super::CompletionModel;

/// A [`CompletionModel`] backed by an `openjiuwen_foundation` model client.
#[derive(Clone)]
pub struct FoundationCompletionModel {
    /// The client every completion is sent to.
    model: Arc<dyn Model>,
    /// An optional system message prepended to the prompt.
    system: Option<String>,
    /// Sampling temperature forwarded to the client.
    temperature: Option<f32>,
    /// Generated-token cap forwarded to the client.
    max_tokens: Option<usize>,
}

impl FoundationCompletionModel {
    /// Wraps `model`.
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            system: None,
            temperature: None,
            max_tokens: None,
        }
    }

    /// Prepends `system` as a system message on every call.
    #[must_use]
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Forwards a sampling temperature.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Forwards a generated-token cap.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// The wrapped client.
    #[must_use]
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }
}

impl fmt::Debug for FoundationCompletionModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FoundationCompletionModel")
            .field("model", &self.model.model_name())
            .field("system", &self.system)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

#[async_trait]
impl CompletionModel for FoundationCompletionModel {
    fn name(&self) -> &str {
        self.model.model_name()
    }

    async fn complete(&self, prompt: &str) -> Result<String> {
        let (messages, request) = build_request(
            self.model.model_name(),
            prompt,
            self.system.as_deref(),
            self.temperature,
            self.max_tokens,
        );
        let response = self.model.invoke(&messages, &request).await?;
        let Some(message) = response.message() else {
            return Err(Error::llm(format!(
                "model `{}` returned no choices",
                self.model.model_name()
            )));
        };
        Ok(message.text().into_owned())
    }
}

/// Builds the messages and request configuration for one completion.
fn build_request(
    model_name: &str,
    prompt: &str,
    system: Option<&str>,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
) -> (Vec<BaseMessage>, ModelRequestConfig) {
    let mut messages = Vec::with_capacity(2);
    if let Some(system) = system.filter(|text| !text.trim().is_empty()) {
        messages.push(BaseMessage::system(system.to_string()));
    }
    messages.push(BaseMessage::user(prompt.to_string()));
    let mut request = ModelRequestConfig::default().with_model_name(model_name);
    if let Some(temperature) = temperature {
        request = request.with_temperature(temperature);
    }
    if let Some(max_tokens) = max_tokens {
        request = request.with_max_tokens(u32::try_from(max_tokens).unwrap_or(u32::MAX));
    }
    (messages, request)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_foundation::Role;

    use super::*;

    #[test]
    fn request_builds_a_user_message_and_forwards_settings() {
        let (messages, request) = build_request("gpt-4o-mini", "hello", None, Some(0.1), Some(256));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[0].text(), "hello");
        assert_eq!(request.model_name, "gpt-4o-mini");
        assert_eq!(request.temperature, Some(0.1));
        assert_eq!(request.max_tokens, Some(256));
        assert!(!request.stream);
    }

    #[test]
    fn request_prepends_a_non_blank_system_message() {
        let (messages, _) = build_request("m", "prompt", Some("be brief"), None, None);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].text(), "be brief");
        assert_eq!(messages[1].role, Role::User);
        let (messages, _) = build_request("m", "prompt", Some("   "), None, None);
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn request_clamps_an_oversized_token_cap() {
        let (_, request) = build_request("m", "prompt", None, None, Some(usize::MAX));
        assert_eq!(request.max_tokens, Some(u32::MAX));
    }
}
