//! An in-process scripted [`Model`] for agent and workflow tests.
//!
//! Ports `tests/unit_tests/fixtures/mock_llm.py::MockLLMModel`: an ordered
//! queue of scripted turns with no prompt matching, a reusable fallback turn
//! once the queue is exhausted, and a call history recorded at call time (so
//! `call_count` and `call_history` behave like the Python fake even before a
//! stream is polled).
//!
//! A turn is stated once as a [`ScriptedTurn`] and projected both to a
//! non-streaming [`ChatResponse`] and to a [`StreamChunk`] sequence, so
//! `invoke` and `stream` read the same script:
//!
//! ```
//! use openjiuwen_foundation::llm::Model;
//! use openjiuwen_test_support::ScriptedModel;
//!
//! let model = ScriptedModel::builder().respond_text("hello").build();
//! assert_eq!(model.provider(), "scripted");
//! assert_eq!(model.model_name(), "scripted-model");
//! assert_eq!(model.remaining(), 1);
//! ```
//!
//! Use [`ScriptedModel`] to fake the [`Model`] trait in agent, workflow, and
//! graph tests; use `MockLlmServer` when the client transport itself is under
//! test.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::llm::{
    ChatResponse, Choice, Model, ModelRequestConfig, StreamAccumulator, StreamChunk,
};
use openjiuwen_foundation::prompt::BaseMessage;

use crate::internal::{lock, DEFAULT_FALLBACK_TEXT};
use crate::responses::{text_chunks, MockCompletion};

/// The default provider name, chosen so a scripted model is obvious in logs.
const DEFAULT_PROVIDER: &str = "scripted";
/// The default model name; also what an unfilled [`ChatResponse::model`] gets.
const DEFAULT_MODEL_NAME: &str = "scripted-model";

/// A failure a scripted turn can produce, with the error category to raise.
///
/// The category is kept separate from the message so a test can assert that a
/// component maps provider failures to the right
/// [`ErrorKind`] without matching on message text.
#[derive(Debug, Clone)]
pub struct ScriptedError {
    /// Category the raised [`Error`] carries.
    pub kind: ErrorKind,
    /// Human-readable failure description.
    pub message: String,
}

impl ScriptedError {
    /// Creates an [`ErrorKind::Llm`] failure, the ordinary provider error.
    #[must_use]
    pub fn llm(message: impl Into<String>) -> Self {
        Self::with_kind(ErrorKind::Llm, message)
    }

    /// Creates an [`ErrorKind::Config`] failure.
    #[must_use]
    pub fn config(message: impl Into<String>) -> Self {
        Self::with_kind(ErrorKind::Config, message)
    }

    /// Creates an [`ErrorKind::Validation`] failure.
    #[must_use]
    pub fn validation(message: impl Into<String>) -> Self {
        Self::with_kind(ErrorKind::Validation, message)
    }

    /// Creates an [`ErrorKind::Transport`] failure, for connection-style errors.
    #[must_use]
    pub fn transport(message: impl Into<String>) -> Self {
        Self::with_kind(ErrorKind::Transport, message)
    }

    /// Creates an [`ErrorKind::Timeout`] failure.
    #[must_use]
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::with_kind(ErrorKind::Timeout, message)
    }

    /// Creates a failure with an arbitrary category.
    #[must_use]
    pub fn with_kind(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Builds the [`Error`] this failure describes.
    ///
    /// Every category with a dedicated message constructor keeps the message;
    /// [`ErrorKind::Cancelled`] ignores it (the Rust error is payload-free) and
    /// [`ErrorKind::External`] wraps it in an [`std::io::Error`].
    pub fn into_error(self) -> Error {
        let Self { kind, message } = self;
        match kind {
            ErrorKind::Config => Error::config(message),
            ErrorKind::Validation => Error::validation(message),
            ErrorKind::Llm => Error::llm(message),
            ErrorKind::Tool => Error::tool(message),
            ErrorKind::Workflow => Error::workflow(message),
            ErrorKind::Graph => Error::graph(message),
            ErrorKind::Session => Error::session(message),
            ErrorKind::Memory => Error::memory(message),
            ErrorKind::Retrieval => Error::retrieval(message),
            ErrorKind::SysOp => Error::sys_op(message),
            ErrorKind::Security => Error::security(message),
            ErrorKind::Mcp => Error::mcp(message),
            ErrorKind::Transport => Error::transport(message),
            ErrorKind::Timeout => Error::timeout(message),
            ErrorKind::NotFound => Error::not_found(message),
            ErrorKind::Unsupported => Error::unsupported(message),
            ErrorKind::Cancelled => Error::cancelled(),
            ErrorKind::External => Error::external(std::io::Error::other(message)),
        }
    }
}

/// One scripted model turn: a complete response, a streamed frame sequence, or
/// a failure.
#[derive(Debug, Clone)]
pub enum ScriptedTurn {
    /// A whole response; `invoke` fills in an empty id and model, `stream`
    /// projects it to frames with [`MockCompletion::to_chunks`].
    Response(ChatResponse),
    /// An explicit frame sequence; `invoke` folds it with
    /// [`StreamAccumulator`], `stream` emits it verbatim.
    Chunks(Vec<StreamChunk>),
    /// A failure; `stream` yields exactly one `Err` item.
    Fail(ScriptedError),
}

impl ScriptedTurn {
    /// Creates a plain text response turn.
    ///
    /// The response carries neither an id nor a model, so both are filled when
    /// the turn is consumed.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self::Response(MockCompletion::text(content).to_chat_response(""))
    }

    /// Creates a turn from an explicit frame sequence.
    #[must_use]
    pub fn chunks(chunks: Vec<StreamChunk>) -> Self {
        Self::Chunks(chunks)
    }

    /// Creates an [`ErrorKind::Llm`] failure turn.
    #[must_use]
    pub fn fail(message: impl Into<String>) -> Self {
        Self::Fail(ScriptedError::llm(message))
    }

    /// Creates a failure turn with an explicit category and message.
    #[must_use]
    pub fn fail_with(error: ScriptedError) -> Self {
        Self::Fail(error)
    }
}

/// How a model call was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallMode {
    /// A non-streaming [`Model::invoke`] call.
    Invoke,
    /// A streaming [`Model::stream`] call.
    Stream,
}

/// One recorded model call: the mode, the conversation, and the request.
#[derive(Debug, Clone)]
pub struct ModelCall {
    /// Which entry point the caller used.
    pub mode: CallMode,
    /// Snapshot of the messages passed to the call.
    pub messages: Vec<BaseMessage>,
    /// Snapshot of the request configuration.
    pub request: ModelRequestConfig,
}

impl ModelCall {
    /// The text of the last user message, or `None` when the call had none.
    #[must_use]
    pub fn last_user_text(&self) -> Option<String> {
        self.messages
            .iter()
            .rev()
            .find(|message| message.role.is_user())
            .map(|message| message.text().into_owned())
    }
}

/// Shared, thread-safe state of a [`ScriptedModel`] and its clones.
struct Inner {
    /// Provider name echoed by [`Model::provider`].
    provider: String,
    /// Model name echoed by [`Model::model_name`].
    model_name: String,
    /// Queued turns, consumed front to back.
    script: Mutex<VecDeque<ScriptedTurn>>,
    /// Turn served once the queue is empty; cloned, never consumed.
    fallback: Mutex<ScriptedTurn>,
    /// Call history in call order.
    calls: Mutex<Vec<ModelCall>>,
    /// Counts materialized turns so ids stay deterministic.
    counter: AtomicUsize,
    /// Wakes [`ScriptedModel::wait_for_calls`] waiters after every call.
    notify: tokio::sync::Notify,
}

/// A [`Model`] that answers from a script.
///
/// Cloning is cheap and shares one script, fallback, and call history, so a
/// test can hand a clone to a spawned task and still inspect the calls from the
/// original. Every answer is deterministic: ids come from a counter
/// (`scripted-1`, `scripted-2`, ...), never from a clock or randomness.
pub struct ScriptedModel {
    /// Shared state behind the (cheap) clone.
    inner: Arc<Inner>,
}

impl Clone for ScriptedModel {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl ScriptedModel {
    /// Creates a builder preloaded with the default provider, model, and
    /// fallback.
    #[must_use]
    pub fn builder() -> ScriptedModelBuilder {
        ScriptedModelBuilder::default()
    }

    /// Creates a model with `turns` queued and the Python defaults otherwise.
    #[must_use]
    pub fn new(turns: impl IntoIterator<Item = ScriptedTurn>) -> Self {
        let model = Self::builder().build();
        model.push_many(turns);
        model
    }

    /// Creates a one-turn text model, the common case for a single call.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self::new([ScriptedTurn::text(content)])
    }

    /// Appends one turn to the end of the queue.
    pub fn push(&self, turn: ScriptedTurn) {
        lock(&self.inner.script).push_back(turn);
    }

    /// Appends several turns in iteration order.
    pub fn push_many(&self, turns: impl IntoIterator<Item = ScriptedTurn>) {
        lock(&self.inner.script).extend(turns);
    }

    /// Replaces the turn served once the queue is empty.
    pub fn set_fallback(&self, turn: ScriptedTurn) {
        *lock(&self.inner.fallback) = turn;
    }

    /// How many scripted turns are still queued.
    #[must_use]
    pub fn remaining(&self) -> usize {
        lock(&self.inner.script).len()
    }

    /// How many calls have been recorded so far.
    #[must_use]
    pub fn call_count(&self) -> usize {
        lock(&self.inner.calls).len()
    }

    /// A snapshot of every recorded call, oldest first.
    #[must_use]
    pub fn calls(&self) -> Vec<ModelCall> {
        lock(&self.inner.calls).clone()
    }

    /// A snapshot of the most recent call, or `None` before the first one.
    #[must_use]
    pub fn last_call(&self) -> Option<ModelCall> {
        lock(&self.inner.calls).last().cloned()
    }

    /// Waits until at least `count` calls have been recorded.
    ///
    /// Python parity: `MockLLMModel` exposes `call_count` for polling; the
    /// notification keeps a concurrent test from sleeping.
    pub async fn wait_for_calls(&self, count: usize) {
        loop {
            // Register before checking so a call landing in between cannot be
            // missed, and `enable` closes the poll-registration window.
            let notified = self.inner.notify.notified();
            tokio::pin!(notified);
            let _ = notified.as_mut().enable();
            if self.call_count() >= count {
                return;
            }
            notified.await;
        }
    }

    /// [`ScriptedModel::wait_for_calls`] with a deadline.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Timeout`] when fewer than `count` calls were
    /// recorded before `timeout` elapsed.
    pub async fn wait_for_calls_timeout(&self, count: usize, timeout: Duration) -> Result<()> {
        match tokio::time::timeout(timeout, self.wait_for_calls(count)).await {
            Ok(()) => Ok(()),
            Err(_elapsed) => Err(Error::timeout(format!(
                "scripted model: expected {count} call(s) within {timeout:?}"
            ))),
        }
    }

    /// Appends a call to the history and wakes any waiter.
    fn record(&self, mode: CallMode, messages: Vec<BaseMessage>, request: ModelRequestConfig) {
        lock(&self.inner.calls).push(ModelCall {
            mode,
            messages,
            request,
        });
        self.inner.notify.notify_waiters();
    }

    /// Removes the next turn, falling back to a clone of the fallback turn.
    fn pop_turn(&self) -> ScriptedTurn {
        match lock(&self.inner.script).pop_front() {
            Some(turn) => turn,
            None => lock(&self.inner.fallback).clone(),
        }
    }

    /// The next deterministic turn number, counting from one.
    fn next_number(&self) -> usize {
        self.inner.counter.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Projects a turn to the non-streaming response `invoke` returns.
    fn materialize(&self, turn: ScriptedTurn) -> Result<ChatResponse> {
        match turn {
            ScriptedTurn::Response(mut response) => {
                let number = self.next_number();
                if response.id.is_empty() {
                    response.id = format!("scripted-{number}");
                }
                if response.model.is_empty() {
                    response.model.clone_from(&self.inner.model_name);
                }
                Ok(response)
            }
            ScriptedTurn::Chunks(chunks) => {
                let number = self.next_number();
                let mut accumulator = StreamAccumulator::new();
                for chunk in &chunks {
                    accumulator.push(chunk);
                }
                let turn = accumulator.finish();
                Ok(ChatResponse {
                    id: format!("scripted-{number}"),
                    model: self.inner.model_name.clone(),
                    choices: vec![Choice {
                        index: 0,
                        message: turn.message,
                        finish_reason: turn.finish_reason,
                    }],
                    usage: turn.usage,
                })
            }
            ScriptedTurn::Fail(error) => Err(error.into_error()),
        }
    }
}

impl Default for ScriptedModel {
    /// A model with an empty script, the default provider/model, and the Python
    /// fallback text.
    fn default() -> Self {
        Self::builder().build()
    }
}

#[async_trait]
impl Model for ScriptedModel {
    fn provider(&self) -> &str {
        self.inner.provider.as_str()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name.as_str()
    }

    async fn invoke(
        &self,
        messages: &[BaseMessage],
        request: &ModelRequestConfig,
    ) -> Result<ChatResponse> {
        self.record(CallMode::Invoke, messages.to_vec(), request.clone());
        self.materialize(self.pop_turn())
    }

    fn stream(
        &self,
        messages: Vec<BaseMessage>,
        request: ModelRequestConfig,
    ) -> BoxStream<'static, Result<StreamChunk>> {
        // Recorded before the returned stream is polled, so `call_count` and
        // `calls` match Python's `call_history` even for an unconsumed stream.
        self.record(CallMode::Stream, messages, request);
        let items: Vec<Result<StreamChunk>> = match self.pop_turn() {
            ScriptedTurn::Response(response) => MockCompletion::from(response)
                .to_chunks()
                .into_iter()
                .map(Ok)
                .collect(),
            ScriptedTurn::Chunks(chunks) => chunks.into_iter().map(Ok).collect(),
            // Exactly one error item and nothing else.
            ScriptedTurn::Fail(error) => vec![Err(error.into_error())],
        };
        Box::pin(futures::stream::iter(items))
    }
}

/// Builder for [`ScriptedModel`].
///
/// Every method consumes and returns the builder, so a model is configured in
/// one expression; [`ScriptedModelBuilder::build`] produces the shared model.
pub struct ScriptedModelBuilder {
    /// Provider name for the built model.
    provider: String,
    /// Model name for the built model.
    model_name: String,
    /// Queued turns, in the order they were added.
    script: VecDeque<ScriptedTurn>,
    /// Reused turn once the queue drains.
    fallback: ScriptedTurn,
}

impl Default for ScriptedModelBuilder {
    /// The Python defaults: provider `"scripted"`, model `"scripted-model"`, an
    /// empty script, and the `"Default mock response"` fallback.
    fn default() -> Self {
        Self {
            provider: DEFAULT_PROVIDER.to_owned(),
            model_name: DEFAULT_MODEL_NAME.to_owned(),
            script: VecDeque::new(),
            fallback: ScriptedTurn::text(DEFAULT_FALLBACK_TEXT),
        }
    }
}

impl ScriptedModelBuilder {
    /// Sets the provider name.
    #[must_use]
    pub fn provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    /// Sets the model name; it is also what unfilled responses adopt.
    #[must_use]
    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = model_name.into();
        self
    }

    /// Queues a whole response as given.
    #[must_use]
    pub fn respond(mut self, response: ChatResponse) -> Self {
        self.script.push_back(ScriptedTurn::Response(response));
        self
    }

    /// Queues a plain text response.
    ///
    /// The id and model are left empty so the built model fills them in; the
    /// configured model name therefore always wins.
    #[must_use]
    pub fn respond_text(mut self, content: impl Into<String>) -> Self {
        self.script.push_back(ScriptedTurn::Response(
            MockCompletion::text(content).to_chat_response(""),
        ));
        self
    }

    /// Queues a response projected from a [`MockCompletion`].
    #[must_use]
    pub fn completion(mut self, completion: MockCompletion) -> Self {
        self.script
            .push_back(ScriptedTurn::Response(completion.to_chat_response("")));
        self
    }

    /// Queues an explicit frame sequence, emitted verbatim by `stream` and
    /// folded by `invoke`.
    #[must_use]
    pub fn chunks(mut self, chunks: Vec<StreamChunk>) -> Self {
        self.script.push_back(ScriptedTurn::Chunks(chunks));
        self
    }

    /// Queues a single-frame text stream, the common streaming case.
    #[must_use]
    pub fn stream_text(self, content: impl Into<String>) -> Self {
        self.chunks(text_chunks(content))
    }

    /// Queues a failure turn.
    #[must_use]
    pub fn fail(mut self, error: ScriptedError) -> Self {
        self.script.push_back(ScriptedTurn::Fail(error));
        self
    }

    /// Replaces the turn served once the queue is empty.
    #[must_use]
    pub fn fallback(mut self, turn: ScriptedTurn) -> Self {
        self.fallback = turn;
        self
    }

    /// Builds the scripted model.
    #[must_use]
    pub fn build(self) -> ScriptedModel {
        let Self {
            provider,
            model_name,
            script,
            fallback,
        } = self;
        ScriptedModel {
            inner: Arc::new(Inner {
                provider,
                model_name,
                script: Mutex::new(script),
                fallback: Mutex::new(fallback),
                calls: Mutex::new(Vec::new()),
                counter: AtomicUsize::new(0),
                notify: tokio::sync::Notify::new(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::StreamExt;
    use openjiuwen_foundation::llm::{FinishReason, Model, ModelExt};
    use openjiuwen_foundation::prompt::Usage;

    use super::*;

    /// Fills a request with the model name a test asserts on.
    fn request(model_name: &str) -> ModelRequestConfig {
        ModelRequestConfig::new(model_name)
    }

    /// Drains a stream into a vector so items can be asserted on.
    async fn collect(
        mut stream: BoxStream<'static, Result<StreamChunk>>,
    ) -> Vec<Result<StreamChunk>> {
        let mut items = Vec::new();
        while let Some(item) = stream.next().await {
            items.push(item);
        }
        items
    }

    #[test]
    fn defaults_and_builder_configuration() {
        let model = ScriptedModel::default();
        assert_eq!(model.provider(), "scripted");
        assert_eq!(model.model_name(), "scripted-model");
        assert_eq!(model.remaining(), 0);
        assert_eq!(model.call_count(), 0);
        assert!(model.calls().is_empty());
        assert!(model.last_call().is_none());

        let built = ScriptedModel::builder()
            .provider("fake-openai")
            .model_name("gpt-fake")
            .respond_text("a")
            .completion(MockCompletion::text("b"))
            .chunks(text_chunks("c"))
            .fail(ScriptedError::transport("d"))
            .build();
        assert_eq!(built.provider(), "fake-openai");
        assert_eq!(built.model_name(), "gpt-fake");
        assert_eq!(built.remaining(), 4);
    }

    #[tokio::test]
    async fn invoke_pops_turns_in_order_and_fills_id_and_model() {
        let model = ScriptedModel::builder()
            .model_name("scripted-test")
            .respond_text("first")
            .respond_text("second")
            .build();
        let request = request("scripted-test");

        let first = model
            .invoke(&[BaseMessage::user("q1")], &request)
            .await
            .expect("first scripted turn");
        assert_eq!(first.id, "scripted-1");
        assert_eq!(first.model, "scripted-test");
        assert_eq!(first.message().expect("choice").text(), "first");
        assert_eq!(first.finish_reason(), Some(&FinishReason::Stop));
        assert_eq!(first.usage, Usage::default());

        let second = model
            .invoke(&[BaseMessage::user("q2")], &request)
            .await
            .expect("second scripted turn");
        assert_eq!(second.id, "scripted-2");
        assert_eq!(second.message().expect("choice").text(), "second");
        assert_eq!(model.remaining(), 0);
    }

    #[tokio::test]
    async fn invoke_preserves_an_explicit_id_and_model() {
        let response = ChatResponse {
            id: "explicit-id".to_owned(),
            model: "explicit-model".to_owned(),
            choices: vec![Choice {
                index: 0,
                message: BaseMessage::assistant("hi"),
                finish_reason: Some(FinishReason::Length),
            }],
            usage: Usage::new(1, 2),
        };
        let model = ScriptedModel::builder().respond(response.clone()).build();

        let received = model
            .invoke(&[], &request("ignored"))
            .await
            .expect("scripted response");
        assert_eq!(received, response);
    }

    #[tokio::test]
    async fn invoke_folds_a_chunk_turn_into_one_response() {
        let completion = MockCompletion::text("streamed ")
            .with_reasoning("why")
            .with_usage(Usage::new(3, 4))
            .with_finish_reason(FinishReason::Length);
        let model = ScriptedModel::builder()
            .model_name("fold-model")
            .chunks(completion.to_chunks())
            .build();

        let response = model
            .invoke(&[BaseMessage::user("q")], &request("fold-model"))
            .await
            .expect("folded chunks");
        assert_eq!(response.id, "scripted-1");
        assert_eq!(response.model, "fold-model");
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.message.text(), "streamed ");
        assert_eq!(choice.message.reasoning_content.as_deref(), Some("why"));
        assert_eq!(choice.finish_reason, Some(FinishReason::Length));
        assert_eq!(response.usage, Usage::new(3, 4));
    }

    #[tokio::test]
    async fn fail_turns_propagate_kind_and_message_on_both_entry_points() {
        let cases = [
            (
                ScriptedError::llm("model rejected the request"),
                ErrorKind::Llm,
            ),
            (
                ScriptedError::transport("connection reset"),
                ErrorKind::Transport,
            ),
            (
                ScriptedError::timeout("first token timed out"),
                ErrorKind::Timeout,
            ),
        ];
        for (error, kind) in cases {
            let model = ScriptedModel::builder()
                .fail(error.clone())
                .fail(error.clone())
                .build();

            let invoke_error = model
                .invoke(&[], &request("scripted-model"))
                .await
                .expect_err("invoke fails");
            assert_eq!(invoke_error.kind(), kind);
            assert!(invoke_error.message().contains(error.message.as_str()));

            let items = collect(model.stream(Vec::new(), request("scripted-model"))).await;
            assert_eq!(items.len(), 1, "a failed stream yields one item");
            let stream_error = items[0].as_ref().expect_err("stream item is an error");
            assert_eq!(stream_error.kind(), kind);
            assert!(stream_error.message().contains(error.message.as_str()));
        }
    }

    #[tokio::test]
    async fn stream_on_a_response_matches_the_chunk_projection() {
        let completion = MockCompletion::text("hello").with_usage(Usage::new(1, 2));
        let expected = completion.to_chunks();
        let model = ScriptedModel::builder()
            .respond(completion.to_chat_response("wire-model"))
            .build();

        let items = collect(model.stream(Vec::new(), request("wire-model"))).await;
        let chunks: Vec<StreamChunk> = items
            .into_iter()
            .map(|item| item.expect("ok frame"))
            .collect();
        assert_eq!(chunks, expected);
        assert_eq!(chunks.last(), Some(&StreamChunk::Done));
    }

    #[tokio::test]
    async fn stream_emits_explicit_chunks_verbatim() {
        let chunks = text_chunks("verbatim");
        let model = ScriptedModel::builder().chunks(chunks.clone()).build();

        let items = collect(model.stream(Vec::new(), request("scripted-model"))).await;
        let received: Vec<StreamChunk> = items
            .into_iter()
            .map(|item| item.expect("ok frame"))
            .collect();
        assert_eq!(received, chunks);
    }

    #[tokio::test]
    async fn calls_record_the_mode_messages_and_request() {
        let model = ScriptedModel::builder()
            .respond_text("a")
            .respond_text("b")
            .build();
        let invocation = ModelRequestConfig::new("rec-model").with_temperature(0.25);

        model
            .invoke(
                &[BaseMessage::system("sys"), BaseMessage::user("hello there")],
                &invocation,
            )
            .await
            .expect("first turn");
        assert_eq!(model.call_count(), 1);
        let call = model.last_call().expect("recorded call");
        assert_eq!(call.mode, CallMode::Invoke);
        assert_eq!(call.messages.len(), 2);
        assert_eq!(call.request, invocation);
        assert_eq!(call.request.model_name, "rec-model");
        assert_eq!(call.last_user_text().as_deref(), Some("hello there"));

        // Recorded at call time, before the returned stream is polled.
        let stream = model.stream(vec![BaseMessage::user("streamed")], invocation.clone());
        assert_eq!(model.call_count(), 2);
        let calls = model.calls();
        assert_eq!(calls[1].mode, CallMode::Stream);
        assert_eq!(calls[1].last_user_text().as_deref(), Some("streamed"));
        assert!(!calls[1].request.stream, "the request is recorded as given");
        drop(stream);
    }

    #[test]
    fn last_user_text_is_none_without_a_user_message() {
        let call = ModelCall {
            mode: CallMode::Invoke,
            messages: vec![BaseMessage::system("sys"), BaseMessage::assistant("hi")],
            request: ModelRequestConfig::default(),
        };
        assert!(call.last_user_text().is_none());
        assert!(ModelCall {
            mode: CallMode::Stream,
            messages: Vec::new(),
            request: ModelRequestConfig::default(),
        }
        .last_user_text()
        .is_none());
    }

    #[tokio::test]
    async fn queue_exhaustion_reuses_the_fallback_without_consuming_it() {
        let model = ScriptedModel::builder().respond_text("only turn").build();
        assert_eq!(model.remaining(), 1);

        let first = model
            .invoke(&[], &request("scripted-model"))
            .await
            .expect("scripted turn");
        assert_eq!(first.message().expect("choice").text(), "only turn");
        assert_eq!(model.remaining(), 0);
        let mut ids = vec![first.id];

        for _ in 0..2 {
            assert_eq!(model.remaining(), 0);
            let fallback = model
                .invoke(&[], &request("scripted-model"))
                .await
                .expect("fallback turn");
            assert_eq!(
                fallback.message().expect("choice").text(),
                "Default mock response"
            );
            assert_eq!(fallback.model, "scripted-model");
            ids.push(fallback.id);
        }
        // The counter still advances once per materialized turn.
        assert_eq!(ids, vec!["scripted-1", "scripted-2", "scripted-3"]);
        assert_eq!(model.call_count(), 3);
    }

    #[tokio::test]
    async fn fallback_is_assignable_and_streams_too() {
        let model = ScriptedModel::new(Vec::<ScriptedTurn>::new());
        model.set_fallback(ScriptedTurn::chunks(text_chunks("fallback text")));

        let response = model
            .invoke(&[], &request("scripted-model"))
            .await
            .expect("fallback chunk turn");
        assert_eq!(response.message().expect("choice").text(), "fallback text");
        assert_eq!(response.id, "scripted-1");

        let items = collect(model.stream(Vec::new(), request("scripted-model"))).await;
        let chunks: Vec<StreamChunk> = items
            .into_iter()
            .map(|item| item.expect("ok frame"))
            .collect();
        assert_eq!(chunks, text_chunks("fallback text"));
    }

    #[tokio::test]
    async fn wait_for_calls_returns_once_the_call_is_recorded() {
        let model = ScriptedModel::text("hi");
        let background = model.clone();
        let handle = tokio::spawn(async move {
            background
                .invoke(&[BaseMessage::user("q")], &ModelRequestConfig::default())
                .await
        });

        model.wait_for_calls(1).await;
        assert_eq!(model.call_count(), 1);
        handle.await.expect("task joins").expect("scripted turn");
    }

    #[tokio::test]
    async fn wait_for_calls_timeout_reports_the_expected_count() {
        let model = ScriptedModel::text("hi");
        model
            .invoke(&[], &ModelRequestConfig::default())
            .await
            .expect("first turn");

        let error = model
            .wait_for_calls_timeout(2, Duration::from_millis(20))
            .await
            .expect_err("the second call never happens");
        assert_eq!(error.kind(), ErrorKind::Timeout);
        let message = error.message().into_owned();
        assert!(message.contains('2'), "{message}");
        assert!(message.contains("20ms"), "{message}");

        model
            .wait_for_calls_timeout(1, Duration::from_millis(20))
            .await
            .expect("the first call is already recorded");
    }

    #[tokio::test]
    async fn complete_through_a_dyn_model_accumulates_and_records_a_stream() {
        let scripted = ScriptedModel::builder()
            .chunks(
                MockCompletion::text("hello")
                    .with_usage(Usage::new(2, 3))
                    .with_finish_reason(FinishReason::Stop)
                    .to_chunks(),
            )
            .build();
        let model: Arc<dyn Model> = Arc::new(scripted.clone());

        let turn = model
            .complete(&[BaseMessage::user("q")], &request("scripted-model"))
            .await
            .expect("completed stream");
        assert_eq!(turn.message.text(), "hello");
        assert_eq!(turn.finish_reason, Some(FinishReason::Stop));
        assert_eq!(turn.usage, Usage::new(2, 3));

        let call = scripted.last_call().expect("recorded call");
        assert_eq!(call.mode, CallMode::Stream);
        assert!(
            call.request.stream,
            "`complete` clones the request with stream = true"
        );
        assert_eq!(call.request.model_name, "scripted-model");
        assert_eq!(call.last_user_text().as_deref(), Some("q"));
    }

    #[test]
    fn into_error_maps_every_kind() {
        let kinds = [
            ErrorKind::Config,
            ErrorKind::Validation,
            ErrorKind::Llm,
            ErrorKind::Tool,
            ErrorKind::Workflow,
            ErrorKind::Graph,
            ErrorKind::Session,
            ErrorKind::Memory,
            ErrorKind::Retrieval,
            ErrorKind::SysOp,
            ErrorKind::Security,
            ErrorKind::Mcp,
            ErrorKind::Transport,
            ErrorKind::Timeout,
            ErrorKind::NotFound,
            ErrorKind::Unsupported,
            ErrorKind::Cancelled,
            ErrorKind::External,
        ];
        for kind in kinds {
            let error = ScriptedError::with_kind(kind, "boom").into_error();
            assert_eq!(error.kind(), kind, "{kind:?}");
            if kind == ErrorKind::Cancelled {
                assert_eq!(error.message(), "cancelled");
            } else {
                assert!(
                    error.message().contains("boom"),
                    "{kind:?}: {}",
                    error.message()
                );
            }
        }
    }

    #[test]
    fn error_constructors_pick_the_documented_kind() {
        let cases = [
            (ScriptedError::llm("m"), ErrorKind::Llm),
            (ScriptedError::config("m"), ErrorKind::Config),
            (ScriptedError::validation("m"), ErrorKind::Validation),
            (ScriptedError::transport("m"), ErrorKind::Transport),
            (ScriptedError::timeout("m"), ErrorKind::Timeout),
        ];
        for (error, kind) in cases {
            assert_eq!(error.kind, kind);
            assert_eq!(error.message, "m");
            assert_eq!(error.into_error().kind(), kind);
        }
        match ScriptedTurn::fail("m") {
            ScriptedTurn::Fail(error) => assert_eq!(error.kind, ErrorKind::Llm),
            other => panic!("expected a failure turn, got {other:?}"),
        }
    }
}
