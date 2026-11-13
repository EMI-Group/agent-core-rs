//! Conversation assembly and turn recording for the `llm` component.
//!
//! This is the message half of
//! `openjiuwen/core/workflow/components/llm/llm_comp.py`:
//!
//! - [`LlmComponent::build_messages`] renders the configured templates (the
//!   `system_prompt_template` / `user_prompt_template` / `template_content` precedence
//!   documented on the parent module).
//! - [`LlmComponent::insert_history`] ports `_insert_history_to_system_and_user_prompt`:
//!   the past conversation is spliced between the leading system prefix and the rest, so
//!   the current user prompt stays last.
//! - [`LlmComponent::assemble_messages`] is Python's `_get_model_input`: templates, then
//!   history, then `LLMPromptFormatter.format_prompt`.
//! - [`LlmComponent::record_turn`] ports `_write_user_message_to_context` +
//!   `_write_assistant_message_to_context`: one append of the sent user text and the raw
//!   assistant reply, and only after a successful model call.
//!
//! Both history methods are no-ops when `enable_history` is `false`, and neither reads
//! nor writes the [`crate::MessageHistory`] in that case, so a history-disabled
//! component produces exactly the conversation it produced before history existed.
//!
//! # Deviations from Python
//!
//! - Python keeps the conversation on its per-component `ModelContext`; here it is the
//!   session global lane `__messages__` (see [`crate::context::history`]), so every
//!   history-enabled LLM node on one session shares it.
//! - The recorded user message is the text actually sent, i.e. after
//!   `format_prompt` replaced it with the markdown/JSON instruction.
//! - The assistant entry stores the raw reply, not the shaped output object.
//! - User and assistant entries are appended after a successful model call, so a failed
//!   call records nothing; a streamed run records its accumulated text once at the end.

use std::collections::BTreeMap;

use openjiuwen_foundation::{BaseMessage, PromptTemplate, Role, TemplateContent};
use serde_json::Value;

use super::{format, LlmComponent};
use crate::context::ComponentContext;

impl LlmComponent {
    /// Builds the conversation for `inputs` from the configured templates.
    ///
    /// See the parent module docs for the template precedence rules.
    pub(super) fn build_messages(&self, inputs: &Value) -> Vec<BaseMessage> {
        let values = render_values(inputs);
        match (
            self.config.system_prompt_template.as_deref(),
            self.config.user_prompt_template.as_deref(),
        ) {
            (Some(system), None) => vec![
                BaseMessage::system(render_text(system, &values)),
                BaseMessage::user(""),
            ],
            (None, Some(user)) => vec![BaseMessage::user(render_text(user, &values))],
            (Some(system), Some(user)) => vec![
                BaseMessage::system(render_text(system, &values)),
                BaseMessage::user(render_text(user, &values)),
            ],
            (None, None) => {
                if self.config.template_content.is_empty() {
                    vec![BaseMessage::user("")]
                } else {
                    PromptTemplate::from_messages(self.config.template_content.clone())
                        .format(&values)
                        .to_messages()
                }
            }
        }
    }

    /// Builds the conversation actually sent to the model: templates → history →
    /// format prompt.
    ///
    /// Port of Python's `_get_model_input`, which inserts history *before*
    /// `LLMPromptFormatter.format_prompt` runs; the format instruction therefore
    /// rewrites the current (last) user prompt, never a history entry.
    pub(super) async fn assemble_messages(
        &self,
        inputs: &Value,
        ctx: &ComponentContext,
    ) -> Vec<BaseMessage> {
        let messages = self.insert_history(self.build_messages(inputs), ctx).await;
        format::format_prompt(
            messages,
            self.config.response_format,
            self.config.markdown_instruction.as_deref(),
            self.config.json_instruction.as_deref(),
            self.config.output_config.as_ref(),
        )
    }

    /// Splices the recorded history between the leading system messages and the rest.
    ///
    /// Port of `_insert_history_to_system_and_user_prompt`: `[system…] + history +
    /// [remaining messages]`, so the current user prompt stays last. History messages
    /// keep their roles verbatim. With `enable_history` off the conversation is
    /// returned untouched and the history store is never read.
    async fn insert_history(
        &self,
        messages: Vec<BaseMessage>,
        ctx: &ComponentContext,
    ) -> Vec<BaseMessage> {
        if !self.config.enable_history {
            return messages;
        }
        let history = ctx.history().messages().await;
        if history.is_empty() {
            return messages;
        }
        let mut messages = messages;
        let split = messages
            .iter()
            .position(|message| message.role != Role::System)
            .unwrap_or(messages.len());
        let rest = messages.split_off(split);
        messages.extend(history);
        messages.extend(rest);
        messages
    }

    /// Records the turn that just completed: the sent user text and the raw reply.
    ///
    /// Port of `_write_user_message_to_context` + `_write_assistant_message_to_context`.
    /// A blank user prompt or a blank reply contributes no entry, and the two entries
    /// travel in one [`crate::MessageHistory::append`] call. With `enable_history` off
    /// nothing is read and nothing is written.
    pub(super) async fn record_turn(
        &self,
        messages: &[BaseMessage],
        reply: &str,
        ctx: &ComponentContext,
    ) {
        if !self.config.enable_history {
            return;
        }
        let mut entries = Vec::with_capacity(2);
        if let Some(prompt) = messages
            .iter()
            .rev()
            .find(|message| message.role == Role::User)
            .map(|message| message.text().into_owned())
        {
            if !prompt.is_empty() {
                entries.push(BaseMessage::user(prompt));
            }
        }
        if !reply.is_empty() {
            entries.push(BaseMessage::assistant(reply));
        }
        if entries.is_empty() {
            return;
        }
        ctx.history().append(entries).await;
    }
}

/// The rendering values of `inputs`: its top-level entries plus `"inputs"`.
fn render_values(inputs: &Value) -> BTreeMap<String, Value> {
    let mut values = BTreeMap::new();
    if let Value::Object(object) = inputs {
        values.extend(
            object
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
    }
    values.insert("inputs".to_owned(), inputs.clone());
    values
}

/// Renders `{{name}}` placeholders in one template string.
///
/// An unresolvable placeholder is kept verbatim by [`PromptTemplate`], matching
/// Python's partial rendering.
fn render_text(template: &str, values: &BTreeMap<String, Value>) -> String {
    match PromptTemplate::new(template).format(values).content {
        TemplateContent::Text(text) => text,
        // `PromptTemplate::new` always builds text content; join defensively so a
        // future change cannot panic here.
        TemplateContent::Messages(messages) => messages
            .iter()
            .map(|message| message.text().into_owned())
            .collect::<String>(),
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_pass_by_value,
    clippy::unnecessary_literal_bound
)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use async_trait::async_trait;
    use futures::StreamExt;
    use openjiuwen_foundation::{Model, StreamChunk};
    use openjiuwen_session::{Session, WorkflowSession};
    use openjiuwen_test_support::ScriptedModel;
    use parking_lot::Mutex;
    use serde_json::json;

    use super::*;
    use crate::component::Component;
    use crate::components::llm::{LlmCompConfig, LlmResponseFormat};
    use crate::MessageHistory;

    /// An in-memory [`MessageHistory`] that records every read and write.
    #[derive(Default)]
    struct FakeHistory {
        messages: Mutex<Vec<BaseMessage>>,
        reads: AtomicUsize,
        appends: Mutex<Vec<Vec<BaseMessage>>>,
    }

    impl FakeHistory {
        fn with(messages: Vec<BaseMessage>) -> Self {
            Self {
                messages: Mutex::new(messages),
                ..Self::default()
            }
        }

        fn reads(&self) -> usize {
            self.reads.load(Ordering::SeqCst)
        }

        fn appends(&self) -> Vec<Vec<BaseMessage>> {
            self.appends.lock().clone()
        }
    }

    #[async_trait]
    impl MessageHistory for FakeHistory {
        async fn messages(&self) -> Vec<BaseMessage> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.messages.lock().clone()
        }

        async fn append(&self, messages: Vec<BaseMessage>) {
            self.messages.lock().extend(messages.iter().cloned());
            self.appends.lock().push(messages);
        }

        async fn clear(&self) {
            self.messages.lock().clear();
        }
    }

    fn component(config: LlmCompConfig) -> LlmComponent {
        LlmComponent::new(config).expect("model is set")
    }

    fn context(history: Arc<FakeHistory>) -> ComponentContext {
        let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("llm-messages"));
        ComponentContext::new(session, "llm", "wf").with_history(history)
    }

    fn pairs(messages: &[BaseMessage]) -> Vec<(Role, String)> {
        messages
            .iter()
            .map(|message| (message.role.clone(), message.text().into_owned()))
            .collect()
    }

    fn user(text: &str) -> (Role, String) {
        (Role::User, text.to_owned())
    }

    fn assistant(text: &str) -> (Role, String) {
        (Role::Assistant, text.to_owned())
    }

    fn system(text: &str) -> (Role, String) {
        (Role::System, text.to_owned())
    }

    fn scripted() -> (ScriptedModel, Arc<dyn Model>) {
        let turns = ScriptedModel::builder()
            .respond_text("first answer")
            .respond_text("second answer")
            .build();
        let model: Arc<dyn Model> = Arc::new(turns.clone());
        (turns, model)
    }

    #[tokio::test]
    async fn insert_history_is_a_no_op_and_reads_nothing_when_disabled() {
        let history = Arc::new(FakeHistory::with(vec![
            BaseMessage::user("u1"),
            BaseMessage::assistant("a1"),
        ]));
        let llm = component(LlmCompConfig {
            model: Some(Arc::new(ScriptedModel::text("answer"))),
            user_prompt_template: Some("{{query}}".to_owned()),
            ..LlmCompConfig::default()
        });

        let messages = llm
            .insert_history(
                llm.build_messages(&json!({"query": "now"})),
                &context(Arc::clone(&history)),
            )
            .await;

        assert_eq!(pairs(&messages), vec![user("now")]);
        assert_eq!(history.reads(), 0, "a disabled component must not read");
    }

    #[tokio::test]
    async fn insert_history_splices_between_the_system_prefix_and_the_prompt() {
        let history = Arc::new(FakeHistory::with(vec![
            BaseMessage::user("u1"),
            BaseMessage::assistant("a1"),
        ]));
        let llm = component(LlmCompConfig {
            model: Some(Arc::new(ScriptedModel::text("answer"))),
            system_prompt_template: Some("system".to_owned()),
            user_prompt_template: Some("{{query}}".to_owned()),
            enable_history: true,
            ..LlmCompConfig::default()
        });

        let messages = llm
            .insert_history(
                llm.build_messages(&json!({"query": "now"})),
                &context(Arc::clone(&history)),
            )
            .await;

        assert_eq!(
            pairs(&messages),
            vec![system("system"), user("u1"), assistant("a1"), user("now"),]
        );
        assert_eq!(history.reads(), 1);
    }

    #[tokio::test]
    async fn assemble_messages_rewrites_only_the_current_user_prompt() {
        let history = Arc::new(FakeHistory::with(vec![
            BaseMessage::user("u1"),
            BaseMessage::assistant("a1"),
        ]));
        let llm = component(LlmCompConfig {
            model: Some(Arc::new(ScriptedModel::text("answer"))),
            user_prompt_template: Some("{{query}}".to_owned()),
            response_format: LlmResponseFormat::Markdown,
            enable_history: true,
            ..LlmCompConfig::default()
        });

        let messages = llm
            .assemble_messages(&json!({"query": "now"}), &context(history))
            .await;

        let pairs = pairs(&messages);
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0], user("u1"), "history entries are kept verbatim");
        assert_eq!(pairs[1], assistant("a1"));
        assert_eq!(pairs[2].0, Role::User);
        assert!(
            pairs[2]
                .1
                .contains("Please return the answer in markdown format."),
            "{}",
            pairs[2].1
        );
        assert!(pairs[2].1.ends_with("The question is: now."), "{pairs:?}");
    }

    #[tokio::test]
    async fn record_turn_is_a_no_op_when_disabled() {
        let history = Arc::new(FakeHistory::default());
        let llm = component(LlmCompConfig {
            model: Some(Arc::new(ScriptedModel::text("answer"))),
            user_prompt_template: Some("{{query}}".to_owned()),
            ..LlmCompConfig::default()
        });

        llm.record_turn(
            &[BaseMessage::user("now")],
            "answer",
            &context(Arc::clone(&history)),
        )
        .await;

        assert_eq!(history.appends(), Vec::<Vec<BaseMessage>>::new());
        assert_eq!(history.messages().await, Vec::<BaseMessage>::new());
    }

    // port of openjiuwen/core/workflow/components/llm/llm_comp.py::LLMExecutable._write_user_message_to_context
    #[tokio::test]
    async fn invoke_appends_the_user_and_assistant_turn_once() {
        let (turns, model) = scripted();
        let history = Arc::new(FakeHistory::default());
        let llm = component(LlmCompConfig {
            model: Some(model),
            user_prompt_template: Some("ask {{query}}".to_owned()),
            enable_history: true,
            ..LlmCompConfig::default()
        });

        llm.invoke(json!({"query": "first"}), &context(Arc::clone(&history)))
            .await
            .unwrap();

        assert_eq!(
            history.appends(),
            vec![vec![
                BaseMessage::user("ask first"),
                BaseMessage::assistant("first answer"),
            ]]
        );
        assert_eq!(
            pairs(&history.messages().await),
            vec![user("ask first"), assistant("first answer")]
        );

        // The second turn is assembled from the fake's contents.
        llm.invoke(json!({"query": "second"}), &context(Arc::clone(&history)))
            .await
            .unwrap();

        let calls = turns.calls();
        assert_eq!(
            pairs(&calls[1].messages),
            vec![
                user("ask first"),
                assistant("first answer"),
                user("ask second"),
            ]
        );
        assert_eq!(
            history.messages().await.len(),
            4,
            "one user + one assistant entry per successful turn"
        );
    }

    #[tokio::test]
    async fn invoke_records_nothing_for_a_blank_prompt_or_a_blank_reply() {
        let history = Arc::new(FakeHistory::default());
        let model: Arc<dyn Model> = Arc::new(ScriptedModel::text(""));
        let llm = component(LlmCompConfig {
            model: Some(Arc::clone(&model)),
            enable_history: true,
            ..LlmCompConfig::default()
        });

        // Neither a user template nor template content: the conversation is one empty
        // user message and the model answers with nothing.
        llm.invoke(json!({}), &context(Arc::clone(&history)))
            .await
            .unwrap();
        assert!(
            history.appends().is_empty(),
            "blank prompt and blank reply contribute no entry"
        );

        let prompt_model: Arc<dyn Model> =
            Arc::new(ScriptedModel::builder().respond_text("a").build());
        let with_prompt = component(LlmCompConfig {
            model: Some(prompt_model),
            user_prompt_template: Some("{{query}}".to_owned()),
            enable_history: true,
            ..LlmCompConfig::default()
        });
        with_prompt
            .invoke(json!({"query": "q"}), &context(Arc::clone(&history)))
            .await
            .unwrap();
        assert_eq!(
            history.appends(),
            vec![vec![BaseMessage::user("q"), BaseMessage::assistant("a")]]
        );
    }

    #[tokio::test]
    async fn stream_records_the_accumulated_answer_once() {
        let turns = ScriptedModel::builder()
            .chunks(vec![
                StreamChunk::Delta {
                    index: 0,
                    role: Some(Role::Assistant),
                    content: Some("he".to_owned()),
                    reasoning: None,
                    tool_call: None,
                    finish_reason: None,
                },
                StreamChunk::Delta {
                    index: 0,
                    role: None,
                    content: Some("llo".to_owned()),
                    reasoning: None,
                    tool_call: None,
                    finish_reason: None,
                },
                StreamChunk::Done,
            ])
            .build();
        let history = Arc::new(FakeHistory::default());
        let llm: Arc<dyn Component> = Arc::new(component(LlmCompConfig {
            model: Some(Arc::new(turns.clone())),
            user_prompt_template: Some("{{query}}".to_owned()),
            enable_history: true,
            ..LlmCompConfig::default()
        }));

        let frames: Vec<_> = llm
            .stream(
                json!({"query": "hi"}),
                Arc::new(context(Arc::clone(&history))),
            )
            .collect()
            .await;
        assert_eq!(frames.len(), 2);
        assert_eq!(
            history.appends(),
            vec![vec![
                BaseMessage::user("hi"),
                BaseMessage::assistant("hello")
            ]]
        );
    }
}
