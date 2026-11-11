//! Neutral descriptions of an assistant turn, shared by both fakes.
//!
//! Ports the response factories of `tests/unit_tests/fixtures/mock_llm.py`
//! (`create_text_response`, `create_reasoning_response`,
//! `create_tool_call_response`) into one type that converts to a
//! [`ChatResponse`] for non-streaming calls and to a [`StreamChunk`] sequence
//! for streaming calls, so a scripted turn is stated once instead of being
//! kept in two synchronized scripts.
//!
//! ```
//! use openjiuwen_test_support::MockCompletion;
//!
//! let completion = MockCompletion::text("hello");
//! let response = completion.to_chat_response("mock-model");
//! assert_eq!(response.message().unwrap().text().as_ref(), "hello");
//! assert_eq!(completion.to_chunks().len(), 4, "role, content, finish, done");
//! ```

use openjiuwen_foundation::llm::{
    ChatResponse, Choice, FinishReason, StreamAccumulator, StreamChunk,
};
use openjiuwen_foundation::prompt::{
    BaseMessage, MessageContent, Role, ToolCall, ToolCallDelta, Usage,
};
use serde_json::Value;

use crate::internal::FUNCTION_KIND;

/// Returns `Some(value)` when the string is non-empty.
fn non_empty(value: String) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

/// One tool call requested by a scripted assistant turn.
///
/// `arguments` is compact JSON text, exactly what a provider puts in
/// `tool_calls[].function.arguments`; use [`MockToolCall::with_raw_arguments`]
/// to script deliberately malformed or fragmentary text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockToolCall {
    /// Provider-style call id; defaults to `call_<name>`.
    pub id: String,
    /// Name of the tool the model wants to call.
    pub name: String,
    /// JSON text of the arguments.
    pub arguments: String,
}

impl MockToolCall {
    /// Creates a call with `id` = `call_<name>` and compact JSON `arguments`.
    ///
    /// Python parity: `create_tool_call_response` auto-generates
    /// `mock_call_<tool_name>`; the Rust id drops the `mock_` prefix because the
    /// mock server already owns the `mock` namespace for response ids.
    #[must_use]
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        let name = name.into();
        Self {
            id: format!("call_{name}"),
            name,
            arguments: serde_json::to_string(&arguments).unwrap_or_else(|_| arguments.to_string()),
        }
    }

    /// Sets the provider call id.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Sets the arguments from raw JSON text, bypassing serialization.
    #[must_use]
    pub fn with_raw_arguments(mut self, arguments: impl Into<String>) -> Self {
        self.arguments = arguments.into();
        self
    }
}

/// A scripted assistant turn: content, reasoning, tool calls, and usage.
///
/// One value describes a response in every projection the fakes need: a
/// non-streaming [`ChatResponse`] through [`MockCompletion::to_chat_response`],
/// a streaming frame sequence through [`MockCompletion::to_chunks`], and the
/// inverse of both through [`MockCompletion::from_chunks`] and
/// `MockCompletion::from(ChatResponse)`.
#[derive(Debug, Clone, PartialEq)]
pub struct MockCompletion {
    /// Provider request id; the mock server assigns `chatcmpl-<prefix><seq>`
    /// when absent.
    pub id: Option<String>,
    /// Model name; each projection falls back to its own default when absent.
    pub model: Option<String>,
    /// Assistant text content.
    pub content: Option<String>,
    /// Reasoning/thinking text some providers return alongside the answer.
    pub reasoning: Option<String>,
    /// Tool calls the assistant requests, in order.
    pub tool_calls: Vec<MockToolCall>,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// Token accounting for the turn.
    pub usage: Usage,
}

impl Default for MockCompletion {
    /// An empty text turn that finishes with [`FinishReason::Stop`].
    fn default() -> Self {
        Self {
            id: None,
            model: None,
            content: None,
            reasoning: None,
            tool_calls: Vec::new(),
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        }
    }
}

impl MockCompletion {
    /// Creates a plain text turn that finishes with [`FinishReason::Stop`].
    ///
    /// Python parity: `tests/unit_tests/fixtures/mock_llm.py::create_text_response`.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: Some(content.into()),
            ..Self::default()
        }
    }

    /// Alias of [`MockCompletion::text`] that documents a JSON payload intent.
    ///
    /// The content is not validated: components that parse JSON decide for
    /// themselves whether the scripted text is valid.
    #[must_use]
    pub fn json(content: impl Into<String>) -> Self {
        Self::text(content)
    }

    /// Creates a text turn with reasoning/thinking content.
    ///
    /// Python parity: `create_reasoning_response`.
    #[must_use]
    pub fn reasoning(reasoning: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            content: Some(content.into()),
            reasoning: Some(reasoning.into()),
            ..Self::default()
        }
    }

    /// Creates a turn with one tool call and no text content.
    ///
    /// Python parity: `create_tool_call_response` (which also leaves the content
    /// empty and finishes with `tool_calls`).
    #[must_use]
    pub fn tool_call(name: impl Into<String>, arguments: Value) -> Self {
        Self::tool_calls(vec![MockToolCall::new(name, arguments)])
    }

    /// Creates a turn with the given tool calls and no text content.
    #[must_use]
    pub fn tool_calls(calls: Vec<MockToolCall>) -> Self {
        Self {
            tool_calls: calls,
            finish_reason: FinishReason::ToolCalls,
            ..Self::default()
        }
    }

    /// Sets the provider request id.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Sets the model name.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the token accounting.
    #[must_use]
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = usage;
        self
    }

    /// Sets the terminal reason.
    #[must_use]
    pub fn with_finish_reason(mut self, reason: FinishReason) -> Self {
        self.finish_reason = reason;
        self
    }

    /// Sets the reasoning/thinking content.
    #[must_use]
    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning = Some(reasoning.into());
        self
    }

    /// Appends one tool call.
    pub fn push_tool_call(&mut self, call: MockToolCall) {
        self.tool_calls.push(call);
    }

    /// Returns `true` when the turn requests at least one tool call.
    #[must_use]
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Projects the turn to a non-streaming chat completion.
    ///
    /// The response id is the scripted id or an empty string (the mock server
    /// stamps its own id on the wire); the model falls back to `default_model`.
    /// The single choice carries the assistant message built from the content,
    /// reasoning, and tool calls.
    #[must_use]
    pub fn to_chat_response(&self, default_model: &str) -> ChatResponse {
        ChatResponse {
            id: self.id.clone().unwrap_or_default(),
            model: self
                .model
                .clone()
                .unwrap_or_else(|| default_model.to_owned()),
            choices: vec![Choice {
                index: 0,
                message: BaseMessage {
                    role: Role::Assistant,
                    content: MessageContent::Text(self.content.clone().unwrap_or_default()),
                    tool_calls: self
                        .tool_calls
                        .iter()
                        .enumerate()
                        .map(|(index, call)| ToolCall {
                            id: Some(call.id.clone()),
                            kind: FUNCTION_KIND.to_owned(),
                            name: call.name.clone(),
                            arguments: call.arguments.clone(),
                            index: Some(index as u32),
                        })
                        .collect(),
                    reasoning_content: self.reasoning.clone(),
                    ..BaseMessage::default()
                },
                finish_reason: Some(self.finish_reason.clone()),
            }],
            usage: self.usage.clone(),
        }
    }

    /// Projects the turn to a streaming frame sequence.
    ///
    /// The order is fixed and matches what an OpenAI-compatible provider sends:
    /// a role delta, one content/reasoning delta when either is scripted, one
    /// delta per tool call, a usage frame only for a non-empty usage, a finish
    /// delta carrying the terminal reason, and [`StreamChunk::Done`].
    #[must_use]
    pub fn to_chunks(&self) -> Vec<StreamChunk> {
        let mut chunks = Vec::with_capacity(self.tool_calls.len() + 4);

        chunks.push(StreamChunk::Delta {
            index: 0,
            role: Some(Role::Assistant),
            content: None,
            reasoning: None,
            tool_call: None,
            finish_reason: None,
        });

        if self.content.is_some() || self.reasoning.is_some() {
            chunks.push(StreamChunk::Delta {
                index: 0,
                role: None,
                content: self.content.clone(),
                reasoning: self.reasoning.clone(),
                tool_call: None,
                finish_reason: None,
            });
        }

        for (index, call) in self.tool_calls.iter().enumerate() {
            chunks.push(StreamChunk::Delta {
                index: 0,
                role: None,
                content: None,
                reasoning: None,
                tool_call: Some(ToolCallDelta {
                    index: Some(u32::try_from(index).unwrap_or(u32::MAX)),
                    id: Some(call.id.clone()),
                    name: Some(call.name.clone()),
                    arguments: Some(call.arguments.clone()),
                }),
                finish_reason: None,
            });
        }

        if !self.usage.is_empty() {
            chunks.push(StreamChunk::Usage(self.usage.clone()));
        }

        chunks.push(StreamChunk::Delta {
            index: 0,
            role: None,
            content: None,
            reasoning: None,
            tool_call: None,
            finish_reason: Some(self.finish_reason.clone()),
        });
        chunks.push(StreamChunk::Done);

        chunks
    }

    /// Folds a frame sequence back into a non-streaming chat completion.
    ///
    /// The result's `id` and `model` are empty because the provider owns them;
    /// a completion round-trips through [`MockCompletion::to_chunks`] and this
    /// function up to id, model, and finish-reason defaults.
    #[must_use]
    pub fn from_chunks(chunks: &[StreamChunk]) -> ChatResponse {
        let mut accumulator = StreamAccumulator::new();
        for chunk in chunks {
            accumulator.push(chunk);
        }
        let turn = accumulator.finish();
        ChatResponse {
            id: String::new(),
            model: String::new(),
            choices: vec![Choice {
                index: 0,
                message: turn.message,
                finish_reason: turn.finish_reason,
            }],
            usage: turn.usage,
        }
    }
}

impl From<ChatResponse> for MockCompletion {
    /// Recovers a scripted turn from a non-streaming response.
    ///
    /// Only the first choice is read; a response without choices becomes the
    /// empty default completion but keeps its id, model, and usage. Empty ids,
    /// models, and text become `None` so the projections fall back to their own
    /// defaults, and a tool call without an id gets `call_<name>`.
    fn from(response: ChatResponse) -> Self {
        let ChatResponse {
            id,
            model,
            choices,
            usage,
        } = response;
        let Some(choice) = choices.into_iter().next() else {
            return Self {
                id: non_empty(id),
                model: non_empty(model),
                usage,
                ..Self::default()
            };
        };

        let Choice {
            message,
            finish_reason,
            ..
        } = choice;
        let content = {
            let text = message.text();
            if text.is_empty() {
                None
            } else {
                Some(text.into_owned())
            }
        };
        let tool_calls = message
            .tool_calls
            .into_iter()
            .map(|call| {
                let ToolCall {
                    id,
                    name,
                    arguments,
                    ..
                } = call;
                MockToolCall {
                    id: id.unwrap_or_else(|| format!("call_{name}")),
                    name,
                    arguments,
                }
            })
            .collect();

        Self {
            id: non_empty(id),
            model: non_empty(model),
            content,
            reasoning: message.reasoning_content,
            tool_calls,
            finish_reason: finish_reason.unwrap_or(FinishReason::Stop),
            usage,
        }
    }
}

/// Frames for a plain text turn: [`MockCompletion::text`] then
/// [`MockCompletion::to_chunks`].
#[must_use]
pub fn text_chunks(content: impl Into<String>) -> Vec<StreamChunk> {
    MockCompletion::text(content).to_chunks()
}

/// Frames for a tool-call turn: [`MockCompletion::tool_calls`] then
/// [`MockCompletion::to_chunks`].
#[must_use]
pub fn tool_call_chunks(calls: Vec<MockToolCall>) -> Vec<StreamChunk> {
    MockCompletion::tool_calls(calls).to_chunks()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Python parity: `tests/unit_tests/fixtures/mock_llm.py::create_text_response`.
    #[test]
    fn default_completion_is_empty_and_stops() {
        let completion = MockCompletion::default();
        assert_eq!(completion.finish_reason, FinishReason::Stop);
        assert!(completion.content.is_none());
        assert!(completion.reasoning.is_none());
        assert_eq!(completion.tool_calls, Vec::new());
        assert!(completion.usage.is_empty());

        let response = completion.to_chat_response("default-model");
        assert_eq!(response.id, "");
        assert_eq!(response.model, "default-model");
        let message = response.message().expect("one choice");
        assert_eq!(message.role, Role::Assistant);
        assert_eq!(message.text().as_ref(), "");
        assert_eq!(message.tool_calls, Vec::new());
        assert_eq!(response.finish_reason(), Some(&FinishReason::Stop));

        let wire = serde_json::to_value(&response).expect("a response serializes");
        assert_eq!(wire["choices"][0]["finish_reason"], json!("stop"));
        assert_eq!(wire["choices"][0]["message"]["content"], json!(""));

        // role delta, finish delta, done
        assert_eq!(completion.to_chunks().len(), 3);
    }

    #[test]
    fn text_and_json_constructors_agree() {
        assert_eq!(
            MockCompletion::json("{\"a\":1}"),
            MockCompletion::text("{\"a\":1}")
        );
        let completion = MockCompletion::text("hello");
        assert!(!completion.has_tool_calls());
        assert_eq!(completion.finish_reason, FinishReason::Stop);
    }

    /// Python parity: `create_tool_call_response` sets `type="function"` and the
    /// finish reason `tool_calls`.
    #[test]
    fn to_chat_response_fills_ids_and_builds_tool_calls() {
        let completion = MockCompletion::tool_calls(vec![
            MockToolCall::new("add", json!({"a": 1, "b": 2})),
            MockToolCall::new("lookup", json!({"city": "Paris"})).with_id("call_custom"),
        ])
        .with_reasoning("thinking")
        .with_usage(Usage::new(3, 4));

        let response = completion.to_chat_response("gpt-4o");
        assert_eq!(response.id, "");
        assert_eq!(response.model, "gpt-4o");
        assert_eq!(response.usage, Usage::new(3, 4));
        assert_eq!(response.finish_reason(), Some(&FinishReason::ToolCalls));

        let message = response.message().expect("one choice");
        assert_eq!(message.role, Role::Assistant);
        assert_eq!(message.reasoning_content.as_deref(), Some("thinking"));
        assert_eq!(message.tool_calls.len(), 2);

        let first = &message.tool_calls[0];
        assert_eq!(first.id.as_deref(), Some("call_add"));
        assert_eq!(first.kind, "function");
        assert_eq!(first.name, "add");
        assert_eq!(first.arguments, "{\"a\":1,\"b\":2}");
        assert_eq!(first.index, Some(0));

        let second = &message.tool_calls[1];
        assert_eq!(second.id.as_deref(), Some("call_custom"));
        assert_eq!(second.index, Some(1));

        let explicit = MockCompletion::text("explicit")
            .with_id("chatcmpl-mock")
            .with_model("scripted-model")
            .to_chat_response("gpt-4o");
        assert_eq!(explicit.id, "chatcmpl-mock");
        assert_eq!(explicit.model, "scripted-model");
    }

    #[test]
    fn to_chunks_emits_role_deltas_usage_finish_and_done_in_order() {
        let completion = MockCompletion::tool_calls(vec![
            MockToolCall::new("add", json!({"a": 1})),
            MockToolCall::new("lookup", json!({"city": "Paris"}))
                .with_id("call_custom")
                .with_raw_arguments("{\"city\":\"Paris\"}"),
        ])
        .with_usage(Usage::new(3, 4));

        assert_eq!(
            completion.to_chunks(),
            vec![
                StreamChunk::Delta {
                    index: 0,
                    role: Some(Role::Assistant),
                    content: None,
                    reasoning: None,
                    tool_call: None,
                    finish_reason: None,
                },
                StreamChunk::Delta {
                    index: 0,
                    role: None,
                    content: None,
                    reasoning: None,
                    tool_call: Some(ToolCallDelta {
                        index: Some(0),
                        id: Some("call_add".to_owned()),
                        name: Some("add".to_owned()),
                        arguments: Some("{\"a\":1}".to_owned()),
                    }),
                    finish_reason: None,
                },
                StreamChunk::Delta {
                    index: 0,
                    role: None,
                    content: None,
                    reasoning: None,
                    tool_call: Some(ToolCallDelta {
                        index: Some(1),
                        id: Some("call_custom".to_owned()),
                        name: Some("lookup".to_owned()),
                        arguments: Some("{\"city\":\"Paris\"}".to_owned()),
                    }),
                    finish_reason: None,
                },
                StreamChunk::Usage(Usage::new(3, 4)),
                StreamChunk::Delta {
                    index: 0,
                    role: None,
                    content: None,
                    reasoning: None,
                    tool_call: None,
                    finish_reason: Some(FinishReason::ToolCalls),
                },
                StreamChunk::Done,
            ]
        );
    }

    #[test]
    fn content_and_tool_calls_keep_their_positional_order() {
        let mut completion = MockCompletion::text("hello")
            .with_reasoning("thinking")
            .with_usage(Usage::new(3, 2))
            .with_finish_reason(FinishReason::ToolCalls);
        completion.push_tool_call(MockToolCall::new("add", json!({"a": 1})));
        completion.push_tool_call(MockToolCall::new("lookup", json!({"city": "Paris"})));

        // role, content+reasoning, tool delta, tool delta, usage, finish, done
        assert_eq!(
            completion.to_chunks(),
            vec![
                StreamChunk::Delta {
                    index: 0,
                    role: Some(Role::Assistant),
                    content: None,
                    reasoning: None,
                    tool_call: None,
                    finish_reason: None,
                },
                StreamChunk::Delta {
                    index: 0,
                    role: None,
                    content: Some("hello".to_owned()),
                    reasoning: Some("thinking".to_owned()),
                    tool_call: None,
                    finish_reason: None,
                },
                StreamChunk::Delta {
                    index: 0,
                    role: None,
                    content: None,
                    reasoning: None,
                    tool_call: Some(ToolCallDelta {
                        index: Some(0),
                        id: Some("call_add".to_owned()),
                        name: Some("add".to_owned()),
                        arguments: Some("{\"a\":1}".to_owned()),
                    }),
                    finish_reason: None,
                },
                StreamChunk::Delta {
                    index: 0,
                    role: None,
                    content: None,
                    reasoning: None,
                    tool_call: Some(ToolCallDelta {
                        index: Some(1),
                        id: Some("call_lookup".to_owned()),
                        name: Some("lookup".to_owned()),
                        arguments: Some("{\"city\":\"Paris\"}".to_owned()),
                    }),
                    finish_reason: None,
                },
                StreamChunk::Usage(Usage::new(3, 2)),
                StreamChunk::Delta {
                    index: 0,
                    role: None,
                    content: None,
                    reasoning: None,
                    tool_call: None,
                    finish_reason: Some(FinishReason::ToolCalls),
                },
                StreamChunk::Done,
            ]
        );

        let chunks = completion.to_chunks();
        assert_eq!(chunks.len(), 7);
        match &chunks[1] {
            StreamChunk::Delta {
                content,
                reasoning,
                tool_call,
                ..
            } => {
                assert_eq!(content.as_deref(), Some("hello"));
                assert_eq!(reasoning.as_deref(), Some("thinking"));
                assert!(
                    tool_call.is_none(),
                    "the content delta must not carry a tool call"
                );
            }
            other => panic!("frame 1 must be the content delta, got {other:?}"),
        }
        assert!(
            chunks[..2].iter().all(|chunk| !matches!(
                chunk,
                StreamChunk::Delta {
                    tool_call: Some(_),
                    ..
                }
            )),
            "no tool-call fragment may precede the content delta"
        );
        assert!(
            chunks[2..4].iter().all(|chunk| chunk.content().is_none()),
            "tool-call deltas must not repeat the text content"
        );
    }

    #[test]
    fn text_and_reasoning_deltas_share_one_frame() {
        let completion = MockCompletion::reasoning("why", "answer");
        assert_eq!(
            completion.to_chunks(),
            vec![
                StreamChunk::Delta {
                    index: 0,
                    role: Some(Role::Assistant),
                    content: None,
                    reasoning: None,
                    tool_call: None,
                    finish_reason: None,
                },
                StreamChunk::Delta {
                    index: 0,
                    role: None,
                    content: Some("answer".to_owned()),
                    reasoning: Some("why".to_owned()),
                    tool_call: None,
                    finish_reason: None,
                },
                StreamChunk::Delta {
                    index: 0,
                    role: None,
                    content: None,
                    reasoning: None,
                    tool_call: None,
                    finish_reason: Some(FinishReason::Stop),
                },
                StreamChunk::Done,
            ]
        );
    }

    #[test]
    fn usage_frame_is_emitted_only_for_a_non_empty_usage() {
        let unmetered = MockCompletion::text("hi").to_chunks();
        assert!(
            unmetered
                .iter()
                .all(|chunk| !matches!(chunk, StreamChunk::Usage(_))),
            "an empty usage must not produce a frame"
        );

        // role, content, usage, finish, done
        let metered = MockCompletion::text("hi")
            .with_usage(Usage::new(3, 4))
            .to_chunks();
        assert_eq!(metered.len(), 5);
        assert_eq!(metered[2], StreamChunk::Usage(Usage::new(3, 4)));
        assert!(matches!(
            metered[3],
            StreamChunk::Delta {
                finish_reason: Some(FinishReason::Stop),
                ..
            }
        ));
        assert_eq!(metered[4], StreamChunk::Done);

        // The usage frame sits after the tool deltas and before the finish delta.
        let tool_call = MockCompletion::tool_call("add", json!({}))
            .with_usage(Usage::new(1, 2))
            .to_chunks();
        assert_eq!(tool_call.len(), 5);
        assert!(matches!(
            tool_call[1],
            StreamChunk::Delta {
                tool_call: Some(_),
                ..
            }
        ));
        assert_eq!(tool_call[2], StreamChunk::Usage(Usage::new(1, 2)));
        assert!(matches!(
            tool_call[3],
            StreamChunk::Delta {
                finish_reason: Some(FinishReason::ToolCalls),
                ..
            }
        ));
        assert_eq!(tool_call[4], StreamChunk::Done);
    }

    #[test]
    fn a_tool_call_completion_round_trips_through_chunks() {
        // Two distinct tool names keep the default ids distinct: two calls with
        // the same `call_<name>` id merge in `ToolCallAccumulator`.
        let completion = MockCompletion::tool_calls(vec![
            MockToolCall::new("add", json!({"a": 1, "b": 2})),
            MockToolCall::new("lookup", json!({"city": "Paris"})),
        ])
        .with_usage(Usage::new(3, 4));

        let mut expected = completion.to_chat_response("m");
        expected.id = String::new();
        expected.model = String::new();

        assert_eq!(
            MockCompletion::from_chunks(&completion.to_chunks()),
            expected
        );
    }

    #[test]
    fn a_text_completion_with_reasoning_round_trips_through_chunks() {
        let completion = MockCompletion::reasoning("why", "answer");
        let mut expected = completion.to_chat_response("m");
        expected.id = String::new();
        expected.model = String::new();

        assert_eq!(
            MockCompletion::from_chunks(&completion.to_chunks()),
            expected
        );
    }

    #[test]
    fn from_chat_response_recovers_content_tool_calls_reasoning_and_usage() {
        let response = ChatResponse {
            id: "chatcmpl-1".to_owned(),
            model: "gpt-4o".to_owned(),
            choices: vec![Choice {
                index: 0,
                message: BaseMessage::assistant("hello")
                    .with_reasoning("why")
                    .with_tool_calls(vec![
                        ToolCall::new("add")
                            .with_arguments(json!({"a": 1}))
                            .with_index(0),
                        ToolCall::new("lookup")
                            .with_id("call_custom")
                            .with_arguments(json!({"city": "Paris"}))
                            .with_index(1),
                    ]),
                finish_reason: Some(FinishReason::ToolCalls),
            }],
            usage: Usage::new(5, 6),
        };

        let completion = MockCompletion::from(response);
        assert_eq!(completion.id.as_deref(), Some("chatcmpl-1"));
        assert_eq!(completion.model.as_deref(), Some("gpt-4o"));
        assert_eq!(completion.content.as_deref(), Some("hello"));
        assert_eq!(completion.reasoning.as_deref(), Some("why"));
        assert_eq!(completion.finish_reason, FinishReason::ToolCalls);
        assert_eq!(completion.usage, Usage::new(5, 6));
        assert_eq!(
            completion.tool_calls,
            vec![
                MockToolCall::new("add", json!({"a": 1})),
                MockToolCall::new("lookup", json!({"city": "Paris"})).with_id("call_custom"),
            ]
        );
    }

    #[test]
    fn from_chat_response_normalizes_empty_fields_and_missing_choices() {
        let empty = MockCompletion::from(ChatResponse {
            id: String::new(),
            model: String::new(),
            choices: vec![Choice {
                index: 0,
                message: BaseMessage::assistant(String::new()),
                finish_reason: None,
            }],
            usage: Usage::default(),
        });
        assert_eq!(empty, MockCompletion::default());

        let no_choice = MockCompletion::from(ChatResponse {
            id: "id".to_owned(),
            model: String::new(),
            choices: Vec::new(),
            usage: Usage::new(1, 1),
        });
        assert_eq!(no_choice.id.as_deref(), Some("id"));
        assert!(no_choice.model.is_none());
        assert_eq!(no_choice.usage, Usage::new(1, 1));
        assert_eq!(no_choice.finish_reason, FinishReason::Stop);
        assert!(no_choice.content.is_none());
    }

    #[test]
    fn builders_append_and_override_the_scripted_turn() {
        let mut completion = MockCompletion::text("hi")
            .with_finish_reason(FinishReason::Length)
            .with_reasoning("because");
        assert!(!completion.has_tool_calls());

        completion.push_tool_call(MockToolCall::new("add", json!({"a": 1})));
        assert!(completion.has_tool_calls());
        assert_eq!(completion.tool_calls.len(), 1);
        assert_eq!(completion.finish_reason, FinishReason::Length);
        assert_eq!(completion.reasoning.as_deref(), Some("because"));
    }

    #[test]
    fn chunk_helpers_match_the_completion_helpers() {
        assert_eq!(
            text_chunks("hello"),
            MockCompletion::text("hello").to_chunks()
        );

        let calls = vec![
            MockToolCall::new("add", json!({"a": 2})),
            MockToolCall::new("lookup", json!({"city": "Paris"})),
        ];
        assert_eq!(
            tool_call_chunks(calls.clone()),
            MockCompletion::tool_calls(calls).to_chunks()
        );
    }
}
