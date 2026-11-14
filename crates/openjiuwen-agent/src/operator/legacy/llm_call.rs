//! `LLMCall`: the legacy model-call wrapper (`openjiuwen/core/operator/legacy/llm_call`).
//!
//! Unlike [`LLMCallOperator`](crate::operator::llm_call::LLMCallOperator) — a
//! tunable prompt handle with no optimizer hook — [`LLMCall`] is a *callable*
//! wrapper: it renders its two prompt templates, appends the caller's history,
//! drives one model call (blocking or streamed), and then reports the exchange
//! to an optional optimizer callback so an evolution loop can observe it.
//!
//! Behaviour preserved from Python's `LLMCall`:
//! - message order is rendered system messages, then `history`, then rendered
//!   user messages ([`LLMCall::invoke`]'s `_format_llm_input`);
//! - an empty constructor user prompt falls back to [`DEFAULT_USER_PROMPT`];
//! - frozen prompts ignore updates silently (the update reports `false`);
//! - the optimizer callback runs exactly once per call, after the model either
//!   answered or its stream drained, and its error propagates to the caller.
//!
//! ```
//! use std::sync::Arc;
//!
//! use openjiuwen_agent::operator::legacy::LLMCall;
//! use openjiuwen_foundation::Model;
//! use openjiuwen_test_support::ScriptedModel;
//! use serde_json::json;
//!
//! async fn demo() -> openjiuwen_common::error::Result<()> {
//!     let model = ScriptedModel::text("hi");
//!     let call = LLMCall::new(
//!         "scripted-model",
//!         Arc::new(model) as Arc<dyn Model>,
//!         "You are {{query}}",
//!         "Answer {{query}}",
//!     );
//!     let message = call.invoke(&json!({"query": "bob"}), None, None, None).await?;
//!     assert_eq!(message.text(), "hi");
//!     Ok(())
//! }
//! ```
use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;

use async_stream::try_stream;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{
    BaseMessage, Model, ModelRequestConfig, PromptTemplate, StreamChunk, TemplateContent,
};
use openjiuwen_session::Session;
use serde_json::Value;

/// User prompt used when [`LLMCall::new`] receives an empty user prompt.
pub const DEFAULT_USER_PROMPT: &str = "{{query}}";

/// Call id used when [`LLMCall::new`] is not told otherwise (Python default).
const DEFAULT_LLM_CALL_ID: &str = "llm_call";

/// One optimizer callback invocation: the call id, its inputs, the response
/// text, and the session the call ran in.
#[derive(Clone)]
pub struct OptimizerTrace {
    /// Id of the [`LLMCall`] that produced this trace.
    pub llm_call_id: String,
    /// Raw inputs the call was rendered with (the caller's JSON value).
    pub inputs: Value,
    /// Accumulated response text (the whole message for `invoke`, the
    /// concatenated deltas for `stream`).
    pub response: String,
    /// Session the call ran in, when the caller supplied one.
    pub session: Option<Arc<dyn Session>>,
}

/// The future an [`OptimizerCallback`] returns; awaited before the call
/// returns, so its error reaches the caller.
pub type OptimizerTraceFuture =
    Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'static>>;

/// Observer of one finished model call (Python's `_optimizer_callback`).
pub type OptimizerCallback = Arc<dyn Fn(OptimizerTrace) -> OptimizerTraceFuture + Send + Sync>;

/// A legacy single-call LLM wrapper with an optimizer trace callback.
///
/// Constructed with a model name and a live [`Model`], the wrapper owns the
/// system/user prompt templates, the two freeze markers, and the callback.
/// Cloning is not provided: an optimizer mutates the handle in place through
/// [`LLMCall::update_system_prompt`] / [`LLMCall::update_user_prompt`].
pub struct LLMCall {
    /// Provider-facing model name sent with every request.
    model_name: String,
    /// Live model the wrapper calls.
    model: Arc<dyn Model>,
    /// System prompt template (rendered into `system` messages).
    system_prompt: PromptTemplate,
    /// User prompt template (rendered into its own messages).
    user_prompt: PromptTemplate,
    /// When `true`, [`LLMCall::update_system_prompt`] is a no-op.
    freeze_system_prompt: bool,
    /// When `true`, [`LLMCall::update_user_prompt`] is a no-op.
    freeze_user_prompt: bool,
    /// Id reported to the optimizer callback.
    llm_call_id: String,
    /// Optional post-call observer.
    optimizer_callback: Option<OptimizerCallback>,
}

impl LLMCall {
    /// Creates a wrapper with the Python defaults.
    ///
    /// `system_prompt` and `user_prompt` each accept a template string or a
    /// message list; an empty user prompt falls back to
    /// [`DEFAULT_USER_PROMPT`]. The system prompt starts evolvable, the user
    /// prompt frozen, and the call id is `"llm_call"`.
    #[must_use]
    pub fn new(
        model_name: impl Into<String>,
        model: Arc<dyn Model>,
        system_prompt: impl Into<TemplateContent>,
        user_prompt: impl Into<TemplateContent>,
    ) -> Self {
        Self {
            model_name: model_name.into(),
            model,
            system_prompt: PromptTemplate::new(system_prompt),
            user_prompt: PromptTemplate::new(with_default_user_prompt(user_prompt.into())),
            freeze_system_prompt: false,
            freeze_user_prompt: true,
            llm_call_id: DEFAULT_LLM_CALL_ID.to_owned(),
            optimizer_callback: None,
        }
    }

    /// Overrides the id reported to the optimizer callback.
    #[must_use]
    pub fn with_llm_call_id(mut self, id: impl Into<String>) -> Self {
        self.llm_call_id = id.into();
        self
    }

    /// Freezes (or unfreezes) the system prompt against updates.
    #[must_use]
    pub const fn with_freeze_system_prompt(mut self, freeze: bool) -> Self {
        self.freeze_system_prompt = freeze;
        self
    }

    /// Freezes (or unfreezes) the user prompt against updates.
    #[must_use]
    pub const fn with_freeze_user_prompt(mut self, freeze: bool) -> Self {
        self.freeze_user_prompt = freeze;
        self
    }

    /// The provider-facing model name sent with every request.
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// The live model this wrapper calls.
    #[must_use]
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    /// The current system prompt template.
    #[must_use]
    pub fn get_system_prompt(&self) -> &PromptTemplate {
        &self.system_prompt
    }

    /// The current user prompt template.
    #[must_use]
    pub fn get_user_prompt(&self) -> &PromptTemplate {
        &self.user_prompt
    }

    /// Replaces the system prompt, reporting whether it changed.
    ///
    /// A frozen system prompt ignores the update and returns `false`.
    pub fn update_system_prompt(&mut self, system_prompt: impl Into<TemplateContent>) -> bool {
        if self.freeze_system_prompt {
            return false;
        }
        self.system_prompt = PromptTemplate::new(system_prompt);
        true
    }

    /// Replaces the user prompt, reporting whether it changed.
    ///
    /// A frozen user prompt ignores the update and returns `false`. Unlike the
    /// constructor, an empty prompt is stored as given (Python's
    /// `update_user_prompt` applies no default).
    pub fn update_user_prompt(&mut self, user_prompt: impl Into<TemplateContent>) -> bool {
        if self.freeze_user_prompt {
            return false;
        }
        self.user_prompt = PromptTemplate::new(user_prompt);
        true
    }

    /// Freezes or unfreezes the system prompt.
    pub const fn set_freeze_system_prompt(&mut self, switch: bool) {
        self.freeze_system_prompt = switch;
    }

    /// Freezes or unfreezes the user prompt.
    pub const fn set_freeze_user_prompt(&mut self, switch: bool) {
        self.freeze_user_prompt = switch;
    }

    /// Whether the system prompt currently ignores updates.
    #[must_use]
    pub const fn get_freeze_system_prompt(&self) -> bool {
        self.freeze_system_prompt
    }

    /// Whether the user prompt currently ignores updates.
    #[must_use]
    pub const fn get_freeze_user_prompt(&self) -> bool {
        self.freeze_user_prompt
    }

    /// The installed optimizer callback, if any.
    #[must_use]
    pub fn get_optimizer_callback(&self) -> Option<&OptimizerCallback> {
        self.optimizer_callback.as_ref()
    }

    /// Installs (or clears, with `None`) the optimizer callback.
    pub fn set_optimizer_callback(&mut self, callback: Option<OptimizerCallback>) {
        self.optimizer_callback = callback;
    }

    /// Renders the prompts, calls the model once, then awaits the optimizer
    /// callback (when installed) before returning the assistant message.
    ///
    /// `history` is inserted between the rendered system and user messages;
    /// `tools` are forwarded as the request's tool definitions.
    ///
    /// # Errors
    ///
    /// Propagates the model's own error, an [`Error::Llm`] when the response
    /// carries no choice, and the optimizer callback's error (the message is
    /// returned only after the callback succeeds).
    pub async fn invoke(
        &self,
        inputs: &Value,
        session: Option<Arc<dyn Session>>,
        history: Option<&[BaseMessage]>,
        tools: Option<&[Value]>,
    ) -> Result<BaseMessage> {
        let messages = self.format_llm_input(inputs, history);
        let request = self.build_request(tools, false);
        let response = self.model.invoke(&messages, &request).await?;
        let message = response
            .message()
            .cloned()
            .ok_or_else(|| Error::llm("model response contained no choice"))?;
        if let Some(callback) = &self.optimizer_callback {
            callback(OptimizerTrace {
                llm_call_id: self.llm_call_id.clone(),
                inputs: inputs.clone(),
                response: message.text().into_owned(),
                session,
            })
            .await?;
        }
        Ok(message)
    }

    /// Streams the model response, accumulating content and awaiting the
    /// optimizer callback exactly once at the end.
    ///
    /// A model error is yielded as one `Err` item and ends the stream before
    /// the callback runs; a callback error is emitted as one final `Err` item.
    /// The request is marked `stream = true`.
    // `history`/`tools` are taken by value because the legacy call signature
    // hands ownership of the conversation and tool definitions to the wrapper,
    // matching Python's `Optional[List[...]]` parameters.
    #[allow(clippy::needless_pass_by_value)]
    pub fn stream(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
        history: Option<Vec<BaseMessage>>,
        tools: Option<Vec<Value>>,
    ) -> BoxStream<'static, Result<StreamChunk>> {
        let messages = self.format_llm_input(&inputs, history.as_deref());
        let request = self.build_request(tools.as_deref(), true);
        let model = Arc::clone(&self.model);
        let callback = self.optimizer_callback.clone();
        let llm_call_id = self.llm_call_id.clone();
        let inner = model.stream(messages, request);

        Box::pin(try_stream! {
            let mut inner = inner;
            let mut collected = String::new();
            while let Some(item) = inner.next().await {
                let chunk = item?;
                if let Some(content) = chunk.content() {
                    collected.push_str(content);
                }
                yield chunk;
            }
            if let Some(callback) = callback {
                callback(OptimizerTrace {
                    llm_call_id,
                    inputs,
                    response: collected,
                    session,
                })
                .await?;
            }
        })
    }

    /// Renders `system` messages, then `history`, then `user` messages.
    fn format_llm_input(
        &self,
        inputs: &Value,
        history: Option<&[BaseMessage]>,
    ) -> Vec<BaseMessage> {
        let values = template_values(inputs);
        let mut messages: Vec<BaseMessage> = self
            .system_prompt
            .format(&values)
            .to_messages()
            .into_iter()
            .map(|message| BaseMessage::system(message.text().into_owned()))
            .collect();
        messages.extend_from_slice(history.unwrap_or(&[]));
        messages.extend(self.user_prompt.format(&values).to_messages());
        messages
    }

    /// Builds the request, forwarding `tools` when the caller supplied them.
    fn build_request(&self, tools: Option<&[Value]>, stream: bool) -> ModelRequestConfig {
        let mut request = ModelRequestConfig::new(self.model_name.clone());
        if let Some(tools) = tools {
            request.tools = Some(Value::Array(tools.to_vec()));
        }
        request.stream = stream;
        request
    }
}

/// The template values of `inputs`: its object entries, or none for a
/// non-object value (Python's `Dict[str, Any]` parameter).
fn template_values(inputs: &Value) -> BTreeMap<String, Value> {
    match inputs {
        Value::Object(map) => map
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) | Value::Array(_) => {
            BTreeMap::new()
        }
    }
}

/// Applies Python's `user_prompt or DEFAULT_USER_PROMPT` to the constructor.
fn with_default_user_prompt(content: TemplateContent) -> TemplateContent {
    let empty = match &content {
        TemplateContent::Text(text) => text.is_empty(),
        TemplateContent::Messages(messages) => messages.is_empty(),
    };
    if empty {
        TemplateContent::Text(DEFAULT_USER_PROMPT.to_owned())
    } else {
        content
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use std::sync::Arc;

    use futures::StreamExt;
    use openjiuwen_common::error::{Error, ErrorKind};
    use openjiuwen_foundation::{BaseMessage, Model};
    use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
    use parking_lot::Mutex;
    use serde_json::json;

    use super::{LLMCall, OptimizerCallback, OptimizerTrace, DEFAULT_USER_PROMPT};

    fn recording_callback() -> (OptimizerCallback, Arc<Mutex<Vec<OptimizerTrace>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&log);
        let callback: OptimizerCallback = Arc::new(move |trace| {
            sink.lock().push(trace);
            Box::pin(async { Ok(()) })
        });
        (callback, log)
    }

    fn text_model(text: &str) -> Arc<ScriptedModel> {
        Arc::new(ScriptedModel::text(text))
    }

    #[tokio::test]
    async fn invoke_orders_system_history_user_and_forwards_tools() {
        let model = text_model("done");
        let call = LLMCall::new(
            "m1",
            Arc::clone(&model) as Arc<dyn Model>,
            "You are {{query}}'s helper",
            "Answer {{query}}",
        );
        let history = vec![
            BaseMessage::user("earlier question"),
            BaseMessage::assistant("earlier answer"),
        ];
        let tools = vec![json!({
            "type": "function",
            "function": {"name": "lookup", "description": "look things up"}
        })];

        let message = call
            .invoke(
                &json!({"query": "alice"}),
                None,
                Some(&history),
                Some(&tools),
            )
            .await
            .expect("model answered");

        assert_eq!(message.text(), "done");
        let recorded = model.last_call().expect("one model call");
        let roles: Vec<&str> = recorded
            .messages
            .iter()
            .map(|message| message.role.as_str())
            .collect();
        assert_eq!(roles, ["system", "user", "assistant", "user"]);
        let texts: Vec<String> = recorded
            .messages
            .iter()
            .map(|message| message.text().into_owned())
            .collect();
        assert_eq!(
            texts,
            [
                "You are alice's helper",
                "earlier question",
                "earlier answer",
                "Answer alice"
            ]
        );
        assert_eq!(recorded.request.model_name, "m1");
        assert_eq!(recorded.request.tools, Some(json!(tools)));
        assert!(!recorded.request.stream);
    }

    #[tokio::test]
    async fn invoke_reports_one_trace_and_returns_only_after_the_callback() {
        let model = text_model("answer text");
        let (callback, log) = recording_callback();
        let mut call = LLMCall::new(
            "m1",
            Arc::clone(&model) as Arc<dyn Model>,
            "sys",
            "Answer {{query}}",
        )
        .with_llm_call_id("call-7");
        call.set_optimizer_callback(Some(callback));

        call.invoke(&json!({"query": "bob"}), None, None, None)
            .await
            .expect("callback accepted the trace");

        let traces = log.lock();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].llm_call_id, "call-7");
        assert_eq!(traces[0].inputs, json!({"query": "bob"}));
        assert_eq!(traces[0].response, "answer text");
        assert!(traces[0].session.is_none());
    }

    #[tokio::test]
    async fn invoke_propagates_a_callback_error() {
        let model = text_model("answer");
        let mut call = LLMCall::new("m1", Arc::clone(&model) as Arc<dyn Model>, "sys", "usr");
        let callback: OptimizerCallback =
            Arc::new(|_trace| Box::pin(async { Err(Error::validation("bad trace")) }));
        call.set_optimizer_callback(Some(callback));

        let error = call
            .invoke(&json!({}), None, None, None)
            .await
            .expect_err("callback error aborts invoke");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(error.message(), "bad trace");
        assert_eq!(model.call_count(), 1);
    }

    #[tokio::test]
    async fn invoke_propagates_a_model_error_without_calling_back() {
        let model = Arc::new(ScriptedModel::new([ScriptedTurn::fail("provider down")]));
        let (callback, log) = recording_callback();
        let mut call = LLMCall::new("m1", Arc::clone(&model) as Arc<dyn Model>, "sys", "usr");
        call.set_optimizer_callback(Some(callback));

        let error = call
            .invoke(&json!({}), None, None, None)
            .await
            .expect_err("model failure surfaces");
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.message().contains("provider down"));
        assert!(log.lock().is_empty());
    }

    #[tokio::test]
    async fn stream_accumulates_deltas_and_traces_once() {
        let model = ScriptedModel::builder().stream_text("done").build();
        let (callback, log) = recording_callback();
        let mut call = LLMCall::new(
            "m1",
            Arc::new(model.clone()) as Arc<dyn Model>,
            "sys",
            "usr",
        )
        .with_llm_call_id("stream-1");
        call.set_optimizer_callback(Some(callback));

        let mut stream = call.stream(json!({"query": "q"}), None, None, None);
        let mut deltas = String::new();
        while let Some(item) = stream.next().await {
            let chunk = item.expect("scripted stream succeeds");
            if let Some(content) = chunk.content() {
                deltas.push_str(content);
            }
        }
        assert_eq!(deltas, "done");

        let traces = log.lock();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].response, "done");
        assert_eq!(traces[0].llm_call_id, "stream-1");
        assert_eq!(traces[0].inputs, json!({"query": "q"}));

        let recorded = model.last_call().expect("one model call");
        assert!(recorded.request.stream);
    }

    #[tokio::test]
    async fn stream_yields_a_model_error_and_skips_the_callback() {
        let model = Arc::new(ScriptedModel::new([ScriptedTurn::fail("stream broke")]));
        let (callback, log) = recording_callback();
        let mut call = LLMCall::new("m1", Arc::clone(&model) as Arc<dyn Model>, "sys", "usr");
        call.set_optimizer_callback(Some(callback));

        let mut stream = call.stream(json!({}), None, None, None);
        let first = stream.next().await.expect("one error item");
        let error = first.expect_err("error surfaces");
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(stream.next().await.is_none());
        assert!(log.lock().is_empty());
    }

    #[tokio::test]
    async fn stream_emits_a_callback_error_as_the_final_item() {
        let model = ScriptedModel::builder().stream_text("done").build();
        let mut call = LLMCall::new("m1", Arc::new(model) as Arc<dyn Model>, "sys", "usr");
        let callback: OptimizerCallback =
            Arc::new(|_trace| Box::pin(async { Err(Error::validation("trace rejected")) }));
        call.set_optimizer_callback(Some(callback));

        let mut stream = call.stream(json!({}), None, None, None);
        let mut items = Vec::new();
        while let Some(item) = stream.next().await {
            items.push(item);
        }
        let (last, leading) = items.split_last().expect("at least one item");
        assert!(leading.iter().all(Result::is_ok));
        let error = last
            .as_ref()
            .expect_err("the callback error is the final item");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(error.message(), "trace rejected");
    }

    #[test]
    fn frozen_prompts_ignore_updates() {
        let mut call = LLMCall::new("m1", text_model("x") as Arc<dyn Model>, "sys", "usr");

        assert!(!call.get_freeze_system_prompt());
        assert!(call.get_freeze_user_prompt());

        assert!(call.update_system_prompt("new sys"));
        assert_eq!(call.get_system_prompt().to_messages()[0].text(), "new sys");
        assert!(!call.update_user_prompt("new usr"));
        assert_eq!(call.get_user_prompt().to_messages()[0].text(), "usr");

        call.set_freeze_system_prompt(true);
        call.set_freeze_user_prompt(false);
        assert!(call.get_freeze_system_prompt());
        assert!(!call.get_freeze_user_prompt());
        assert!(!call.update_system_prompt("ignored"));
        assert_eq!(call.get_system_prompt().to_messages()[0].text(), "new sys");
        assert!(call.update_user_prompt("new usr"));
        assert_eq!(call.get_user_prompt().to_messages()[0].text(), "new usr");
    }

    #[test]
    fn builder_flags_and_accessors_report_their_values() {
        let model = text_model("x");
        let mut call = LLMCall::new("m2", Arc::clone(&model) as Arc<dyn Model>, "sys", "usr")
            .with_llm_call_id("custom-id")
            .with_freeze_system_prompt(true)
            .with_freeze_user_prompt(false);

        assert_eq!(call.model_name(), "m2");
        assert_eq!(call.model().model_name(), "scripted-model");
        assert!(call.get_freeze_system_prompt());
        assert!(!call.get_freeze_user_prompt());

        assert!(call.get_optimizer_callback().is_none());
        let (callback, _log) = recording_callback();
        call.set_optimizer_callback(Some(callback));
        assert!(call.get_optimizer_callback().is_some());
        call.set_optimizer_callback(None);
        assert!(call.get_optimizer_callback().is_none());
    }

    #[test]
    fn empty_user_prompt_falls_back_to_the_default() {
        let empty_text = LLMCall::new("m1", text_model("x") as Arc<dyn Model>, "sys", "");
        assert_eq!(
            empty_text.get_user_prompt().content,
            openjiuwen_foundation::TemplateContent::Text(DEFAULT_USER_PROMPT.to_owned())
        );
        let empty_messages =
            LLMCall::new("m1", text_model("x") as Arc<dyn Model>, "sys", Vec::new());
        assert_eq!(
            empty_messages.get_user_prompt().content,
            openjiuwen_foundation::TemplateContent::Text(DEFAULT_USER_PROMPT.to_owned())
        );
    }

    #[tokio::test]
    async fn non_object_inputs_render_without_substitution() {
        let model = text_model("ok");
        let call = LLMCall::new(
            "m1",
            Arc::clone(&model) as Arc<dyn Model>,
            "sys {{query}}",
            "usr",
        );
        call.invoke(&json!(["not", "an", "object"]), None, None, None)
            .await
            .expect("renders with no values");

        let recorded = model.last_call().expect("one model call");
        assert_eq!(recorded.messages[0].text(), "sys {{query}}");
    }
}
