//! The chat-runner seam: turn a rendered `(system, user)` pair into reply text.
//!
//! Python's `ChatAgent` owns an `LLMCall` and a `Model`; here the execution step
//! sits behind [`ChatRunner`] so a tuning run can be driven by a real model, by a
//! test double, or by a caller's own agent runtime. Two implementations ship:
//!
//! - [`ModelChatRunner`] builds one request per reply — the system message, the
//!   user message, and the [`ChatConfig`]'s model name and sampling settings —
//!   and returns the assistant text.
//! - [`ScriptedChatRunner`] replays canned replies for offline tests, keyed by a
//!   case marker or by call order, and records every `(system, user)` pair.
//!
//! Every runner is stateless with respect to a tuning run: the conversation is
//! assembled by [`ChatAgent`](super::ChatAgent) from the candidate prompt and the
//! case, and a runner only answers it.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};

use super::config::ChatConfig;

/// Answers one rendered chat turn.
///
/// The seam a tuning run is bound to: [`ChatAgent`](super::ChatAgent) renders the
/// candidate prompt into `system` and the case into `user`, and the runner
/// returns the reply text that becomes the answer map's `output` entry.
#[async_trait]
pub trait ChatRunner: Send + Sync {
    /// Answers the `system`/`user` pair and returns the reply text.
    ///
    /// # Errors
    ///
    /// Implementation-defined: a model runner propagates the provider error, a
    /// scripted runner reports an unknown call.
    async fn reply(&self, system: &str, user: &str) -> Result<String>;
}

/// A [`ChatRunner`] that answers one [`Model`] request per reply.
///
/// The request carries the configured model name ([`ChatConfig::model_name`])
/// and the configured sampling settings; when they are absent the provider's own
/// defaults apply. A completion without any choice is reported as
/// [`Error::llm`] instead of degrading to an empty answer, so a broken provider
/// stays visible.
pub struct ModelChatRunner {
    /// The model consulted for every reply.
    model: Arc<dyn Model>,
    /// The request template every reply is sent with.
    request: ModelRequestConfig,
}

impl ModelChatRunner {
    /// Creates a runner over `model`, taking the request's model name from it.
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        let request = ModelRequestConfig::new(model.model_name().to_owned());
        Self { model, request }
    }

    /// Creates a runner over `model` using `config`'s model name and sampling
    /// settings.
    ///
    /// The configuration is expected to be validated already
    /// ([`ChatConfig::validate`]); the sampling values are narrowed to the `f32`
    /// a request carries.
    #[must_use]
    pub fn from_config(config: &ChatConfig, model: Arc<dyn Model>) -> Self {
        let mut request = ModelRequestConfig::new(config.model_name.trim());
        if let Some(temperature) = config.temperature {
            request = request.with_temperature(narrow(temperature));
        }
        if let Some(top_p) = config.top_p {
            request = request.with_top_p(narrow(top_p));
        }
        Self { model, request }
    }

    /// Replaces the request template.
    ///
    /// Useful to set fields [`from_config`](Self::from_config) does not carry,
    /// such as `max_tokens` or a `stop` list.
    #[must_use]
    pub fn with_request(mut self, request: ModelRequestConfig) -> Self {
        self.request = request;
        self
    }

    /// The model this runner consults.
    #[must_use]
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    /// The request template this runner sends.
    #[must_use]
    pub fn request(&self) -> &ModelRequestConfig {
        &self.request
    }
}

impl fmt::Debug for ModelChatRunner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelChatRunner")
            .field("provider", &self.model.provider())
            .field("model_name", &self.model.model_name())
            .field("request_model", &self.request.model_name)
            .field("temperature", &self.request.temperature)
            .field("top_p", &self.request.top_p)
            .finish()
    }
}

#[async_trait]
impl ChatRunner for ModelChatRunner {
    async fn reply(&self, system: &str, user: &str) -> Result<String> {
        let messages = [
            BaseMessage::system(system.to_owned()),
            BaseMessage::user(user.to_owned()),
        ];
        let completion = self.model.invoke(&messages, &self.request).await?;
        let Some(message) = completion.message() else {
            return Err(Error::llm(
                "chat runner model returned no choices; expected a reply message",
            ));
        };
        Ok(message.text().into_owned())
    }
}

/// Narrows a validated sampling setting to the `f32` a request carries.
///
/// Both validated settings live in `0.0..=2.0`, so no significant digit is lost.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn narrow(value: f64) -> f32 {
    value as f32
}

/// One recorded [`ScriptedChatRunner`] call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ChatCall {
    /// The system message that was sent.
    pub system: String,
    /// The user message that was sent.
    pub user: String,
}

/// A [`ChatRunner`] that replays canned replies.
///
/// A reply is resolved in three steps, in order:
///
/// 1. the first marker (in sorted order) that occurs in the user message, then
///    the first that occurs in the system message — so a reply can be keyed by a
///    case's input text or by the candidate prompt;
/// 2. the next reply added with [`with_ordered_reply`](Self::with_ordered_reply),
///    consumed front to back;
/// 3. the reply set with [`with_fallback`](Self::with_fallback).
///
/// An unresolved call reports [`Error::not_found`]. Every call — resolved or not
/// — is recorded, so a test can assert the messages a tuning run produced and in
/// which order.
#[derive(Debug, Default)]
pub struct ScriptedChatRunner {
    /// Replies keyed by marker text.
    replies: BTreeMap<String, String>,
    /// Replies served in call order once no marker matches.
    ordered: Mutex<VecDeque<String>>,
    /// Reply served when neither a marker nor an ordered reply matches.
    fallback: Option<String>,
    /// Calls in call order.
    calls: Mutex<Vec<ChatCall>>,
}

impl ScriptedChatRunner {
    /// Creates a runner without replies.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Answers any call whose user or system message contains `marker` with
    /// `reply`.
    #[must_use]
    pub fn with_reply(mut self, marker: impl Into<String>, reply: impl Into<String>) -> Self {
        self.replies.insert(marker.into(), reply.into());
        self
    }

    /// Queues `reply` to be served, in call order, once no marker matches.
    #[must_use]
    pub fn with_ordered_reply(mut self, reply: impl Into<String>) -> Self {
        self.ordered
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push_back(reply.into());
        self
    }

    /// Answers every unresolved call with `reply`.
    #[must_use]
    pub fn with_fallback(mut self, reply: impl Into<String>) -> Self {
        self.fallback = Some(reply.into());
        self
    }

    /// How many ordered replies are still queued.
    #[must_use]
    pub fn remaining(&self) -> usize {
        lock(&self.ordered).len()
    }

    /// The calls recorded so far, oldest first.
    #[must_use]
    pub fn calls(&self) -> Vec<ChatCall> {
        lock(&self.calls).clone()
    }

    /// How many calls were recorded.
    #[must_use]
    pub fn call_count(&self) -> usize {
        lock(&self.calls).len()
    }

    /// The most recent call, or `None` before the first one.
    #[must_use]
    pub fn last_call(&self) -> Option<ChatCall> {
        lock(&self.calls).last().cloned()
    }

    /// Resolves the reply for one rendered turn.
    fn resolve(&self, system: &str, user: &str) -> Option<String> {
        if let Some(reply) = self
            .replies
            .iter()
            .find(|(marker, _)| user.contains(marker.as_str()))
            .or_else(|| {
                self.replies
                    .iter()
                    .find(|(marker, _)| system.contains(marker.as_str()))
            })
        {
            return Some(reply.1.clone());
        }
        if let Some(reply) = lock(&self.ordered).pop_front() {
            return Some(reply);
        }
        self.fallback.clone()
    }
}

#[async_trait]
impl ChatRunner for ScriptedChatRunner {
    async fn reply(&self, system: &str, user: &str) -> Result<String> {
        lock(&self.calls).push(ChatCall {
            system: system.to_owned(),
            user: user.to_owned(),
        });
        self.resolve(system, user).ok_or_else(|| {
            Error::not_found(format!(
                "scripted chat runner has no reply for user message {user:?}"
            ))
        })
    }
}

/// Locks `mutex`, recovering the guard when a previous holder panicked.
///
/// A poisoned lock only means some other caller panicked; the recorded calls and
/// queued replies are still consistent, so recovering keeps this helper
/// panic-free.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use openjiuwen_test_support::{ScriptedError, ScriptedModel};

    use super::*;

    #[tokio::test]
    async fn a_scripted_runner_answers_by_marker() {
        let runner = ScriptedChatRunner::new()
            .with_reply("capital of France", "Paris")
            .with_reply("2 + 2", "4");
        assert_eq!(
            runner
                .reply("be brief", "capital of France?")
                .await
                .unwrap(),
            "Paris"
        );
        assert_eq!(runner.reply("be brief", "2 + 2 = ?").await.unwrap(), "4");
        assert_eq!(runner.call_count(), 2);
    }

    #[tokio::test]
    async fn a_marker_matches_the_system_message_too() {
        let runner = ScriptedChatRunner::new().with_reply("## Candidate", "from the prompt");
        let reply = runner
            .reply("## Candidate instructions\n\nx", "q")
            .await
            .unwrap();
        assert_eq!(reply, "from the prompt");
    }

    #[tokio::test]
    async fn ordered_replies_are_consumed_in_call_order() {
        let runner = ScriptedChatRunner::new()
            .with_ordered_reply("first")
            .with_ordered_reply("second");
        assert_eq!(runner.remaining(), 2);
        assert_eq!(runner.reply("s", "u1").await.unwrap(), "first");
        assert_eq!(runner.remaining(), 1);
        assert_eq!(runner.reply("s", "u2").await.unwrap(), "second");
        assert_eq!(runner.remaining(), 0);
    }

    #[tokio::test]
    async fn a_marker_wins_over_the_ordered_queue() {
        let runner = ScriptedChatRunner::new()
            .with_reply("pinned", "by marker")
            .with_ordered_reply("by order");
        assert_eq!(
            runner.reply("s", "pinned question").await.unwrap(),
            "by marker"
        );
        assert_eq!(runner.remaining(), 1);
        assert_eq!(
            runner.reply("s", "other question").await.unwrap(),
            "by order"
        );
    }

    #[tokio::test]
    async fn the_fallback_answers_whatever_is_left() {
        let runner = ScriptedChatRunner::new()
            .with_ordered_reply("queued")
            .with_fallback("fallback");
        assert_eq!(runner.reply("s", "u1").await.unwrap(), "queued");
        assert_eq!(runner.reply("s", "u2").await.unwrap(), "fallback");
    }

    #[tokio::test]
    async fn an_unscripted_call_is_not_found_and_still_recorded() {
        let runner = ScriptedChatRunner::new();
        let error = runner.reply("system text", "user text").await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::NotFound);
        assert!(error.to_string().contains("user text"), "{error}");

        let calls = runner.calls();
        assert_eq!(
            calls,
            vec![ChatCall {
                system: "system text".to_owned(),
                user: "user text".to_owned(),
            }]
        );
        assert_eq!(runner.last_call(), calls.first().cloned());
    }

    #[tokio::test]
    async fn recorded_calls_keep_the_conversation_order() {
        let runner = ScriptedChatRunner::new().with_fallback("ok");
        runner.reply("s1", "u1").await.unwrap();
        runner.reply("s2", "u2").await.unwrap();
        let calls = runner.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].system, "s1");
        assert_eq!(calls[0].user, "u1");
        assert_eq!(calls[1].system, "s2");
        assert_eq!(calls[1].user, "u2");
        assert_eq!(runner.last_call().unwrap().system, "s2");
    }

    #[tokio::test]
    async fn a_model_runner_sends_the_configured_request() {
        let model = ScriptedModel::builder()
            .model_name("scripted-name")
            .respond_text("hello")
            .build();
        let config = ChatConfig::new("agent-1", "gpt-4o-mini")
            .with_temperature(0.25)
            .with_top_p(0.75);
        let runner = ModelChatRunner::from_config(&config, Arc::new(model.clone()));
        assert_eq!(runner.request().model_name, "gpt-4o-mini");
        assert_eq!(runner.request().temperature, Some(0.25));
        assert_eq!(runner.request().top_p, Some(0.75));

        let reply = runner.reply("be brief", "2 + 2?").await.unwrap();
        assert_eq!(reply, "hello");

        let calls = model.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].messages.len(), 2);
        assert!(calls[0].messages[0].role.is_system());
        assert_eq!(calls[0].messages[0].text(), "be brief");
        assert!(calls[0].messages[1].role.is_user());
        assert_eq!(calls[0].messages[1].text(), "2 + 2?");
        assert_eq!(calls[0].request.model_name, "gpt-4o-mini");
    }

    #[tokio::test]
    async fn a_model_runner_without_a_config_uses_the_models_own_name() {
        let model = ScriptedModel::builder()
            .model_name("own-name")
            .respond_text("hi")
            .build();
        let runner = ModelChatRunner::new(Arc::new(model.clone()));
        assert_eq!(runner.request().model_name, "own-name");
        assert_eq!(runner.request().temperature, None);
        assert_eq!(runner.model().model_name(), "own-name");
        runner.reply("s", "u").await.unwrap();
        assert_eq!(model.calls()[0].request.model_name, "own-name");
    }

    #[tokio::test]
    async fn a_model_runner_propagates_a_provider_failure() {
        let model = ScriptedModel::builder()
            .fail(ScriptedError::llm("provider down"))
            .build();
        let runner = ModelChatRunner::new(Arc::new(model));
        let error = runner.reply("s", "u").await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Llm);
    }

    #[test]
    fn a_model_runner_is_debug() {
        let model = ScriptedModel::text("hi");
        let config = ChatConfig::new("agent-1", "gpt-4o-mini").with_temperature(0.5);
        let runner = ModelChatRunner::from_config(&config, Arc::new(model));
        let rendered = format!("{runner:?}");
        assert!(rendered.contains("ModelChatRunner"), "{rendered}");
        assert!(rendered.contains("gpt-4o-mini"), "{rendered}");
    }
}
