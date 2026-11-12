//! Feedback-driven prompt rewriting.
//!
//! Ports `openjiuwen.dev_tools.prompt_builder.builder.feedback_prompt_builder`
//! (`FeedbackPromptBuilder`). One [`FeedbackPromptBuilder::build`] call makes
//! exactly two model calls:
//!
//! 1. the **intent-normalization** call renders the bank's `feedback_intent`
//!    template with the original (untagged) prompt and the raw feedback, and
//!    asks the model whether the feedback is actionable and, if so, for a
//!    clearer phrasing of it;
//! 2. the **optimize** call renders the bank's mode template — `general`,
//!    `insert`, or `select` — with that phrasing and returns the rewritten
//!    prompt.
//!
//! The three modes differ only in what the optimize template sees:
//! `general` rewrites the whole prompt, `insert` splices
//! [`INSERT_STR`](crate::prompt_builder::INSERT_STR) into the prompt at
//! [`FeedbackOptions::start_pos`] and asks for content to insert there, and
//! `select` hands the model the full prompt plus the fragment in
//! `start_pos..end_pos` to rewrite.
//!
//! `start_pos` and `end_pos` are **character** indices, matching Python's `str`
//! slicing, so a multi-byte prompt is never corrupted.
//!
//! # Deviations from Python
//!
//! - Validation is strict and happens before any model call: a blank prompt or
//!   feedback, a missing insert position, and a missing or inverted select range
//!   are [`Error::Validation`] instead of a Python `build_error` raised halfway
//!   through template assembly.
//! - When the intent answer cannot be parsed, or marks the feedback as
//!   worthless, or carries no phrasing, the builder logs at `debug` level and
//!   falls back to the raw feedback (Python warns and does the same).
//! - An optimize answer with no choice or with blank text is [`Error::Llm`];
//!   Python returns `response.content`, which may be `None`.
//! - [`FeedbackPromptBuilder::stream_build`] streams only the optimize call; the
//!   intent stage is a non-streamed `invoke`, so a stream always shows only the
//!   final answer. Python streams the optimize call as well but yields every
//!   chunk's content, `None` included; here only non-empty text deltas are
//!   yielded.

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{ChatResponse, Model, ModelRequestConfig, StreamChunk};
use openjiuwen_foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::language::Language;
use super::templates::{bank, user_template};
use super::utils::{insert_marker, parse_feedback_intent, select_range};

/// Which part of the prompt the feedback should change.
///
/// Mirrors Python's `Literal["general", "insert", "select"]` mode argument and
/// serializes with the same `snake_case` tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackMode {
    /// Rewrite the whole prompt; the default, and the only mode that ignores
    /// the positional fields.
    #[default]
    General,
    /// Generate content to insert at [`FeedbackOptions::start_pos`].
    Insert,
    /// Rewrite the fragment in [`FeedbackOptions::start_pos`]`..`[`FeedbackOptions::end_pos`].
    Select,
}

/// Options for one [`FeedbackPromptBuilder::build`] call.
///
/// Groups the Python keyword arguments of `build`/`stream_build` into one value,
/// so adding an option never changes the method signature. Defaults match
/// Python's: no feedback text, [`FeedbackMode::General`], no positions, and
/// [`Language::ZhCn`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct FeedbackOptions {
    /// The user feedback to apply; must not be blank.
    pub feedback: String,
    /// Which rewriting mode to use.
    pub mode: FeedbackMode,
    /// Character index where insert mode splices the marker, or where select
    /// mode's fragment starts.
    pub start_pos: Option<usize>,
    /// Character index one past select mode's fragment; unused by the other
    /// modes.
    pub end_pos: Option<usize>,
    /// Language of the bundled template bank.
    pub language: Language,
}

/// The validated positional requirements of a mode.
///
/// [`validate`] resolves the optional positions into this enum, so assembling
/// the optimize messages never has to re-check an `Option` or fall back to a
/// default index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Placement {
    /// Rewrite the whole prompt.
    General,
    /// Splice the insert marker at this character index.
    Insert(usize),
    /// Rewrite the fragment in this half-open character range.
    Select { start: usize, end: usize },
}

/// Rewrites a prompt from user feedback with a model.
///
/// The model is shared, never rebuilt, so the caller keeps ownership of its
/// configuration.
pub struct FeedbackPromptBuilder {
    /// Model used for the intent and optimize calls.
    model: Arc<dyn Model>,
}

impl FeedbackPromptBuilder {
    /// Creates a builder over `model`.
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self { model }
    }

    /// Rewrites `prompt` according to `options` with exactly two model calls.
    ///
    /// Returns the optimize answer's first choice unchanged.
    ///
    /// # Errors
    ///
    /// - [`Error::Validation`] when the prompt or feedback is blank, insert mode
    ///   lacks `start_pos` or it exceeds the prompt length, or select mode lacks
    ///   a position or its range violates `0 <= start_pos < end_pos <= len`.
    /// - [`Error::Llm`] when the optimize answer has no choice or blank text.
    /// - Any error the model returns, unchanged.
    pub async fn build(&self, prompt: &str, options: &FeedbackOptions) -> Result<String> {
        let messages = optimize_messages(self.model.as_ref(), prompt, options).await?;
        let request = ModelRequestConfig::new(self.model.model_name());
        let response = self.model.invoke(&messages, &request).await?;
        answer_text(&response, self.model.model_name())
    }

    /// Prepares the two stages and streams the optimize answer.
    ///
    /// The intent stage is a non-streamed `invoke`, so the stream always shows
    /// only the final answer. The returned stream owns a clone of the
    /// [`Arc<dyn Model>`], so the builder is not borrowed and may be dropped.
    ///
    /// A validation or intent-model error is yielded as the single `Err` item
    /// and ends the stream. Only non-empty text deltas of the optimize call are
    /// forwarded as `Ok(String)`; usage, done, and empty deltas are skipped, and
    /// a model error is yielded as `Err` before the stream terminates.
    pub fn stream_build(
        &self,
        prompt: String,
        options: FeedbackOptions,
    ) -> BoxStream<'static, Result<String>> {
        let model = Arc::clone(&self.model);

        Box::pin(async_stream::stream! {
            let messages = match optimize_messages(model.as_ref(), &prompt, &options).await {
                Ok(messages) => messages,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            let request = ModelRequestConfig::new(model.model_name());
            let mut stream = model.stream(messages, request);
            while let Some(item) = stream.next().await {
                match item {
                    Ok(StreamChunk::Delta { content: Some(text), .. }) if !text.is_empty() => {
                        yield Ok(text);
                    }
                    Ok(_) => {}
                    Err(error) => {
                        yield Err(error);
                        return;
                    }
                }
            }
        })
    }
}

/// Validates `options` against `prompt` and resolves the mode's positions.
///
/// Runs before any model call, so an invalid request never consumes a turn.
/// Every error names the violated rule.
fn validate(prompt: &str, options: &FeedbackOptions) -> Result<Placement> {
    if prompt.trim().is_empty() {
        return Err(Error::validation("prompt cannot be empty"));
    }
    if options.feedback.trim().is_empty() {
        return Err(Error::validation("feedback cannot be empty"));
    }

    let char_len = prompt.chars().count();
    match options.mode {
        FeedbackMode::General => Ok(Placement::General),
        FeedbackMode::Insert => match options.start_pos {
            None => Err(Error::validation(
                "insert mode requires start_pos, satisfying 0 <= start_pos <= len",
            )),
            Some(start) if start <= char_len => Ok(Placement::Insert(start)),
            Some(start) => Err(Error::validation(format!(
                "insert start_pos {start} is out of bounds for a prompt of {char_len} \
                 characters; requires 0 <= start_pos <= {char_len}"
            ))),
        },
        FeedbackMode::Select => match (options.start_pos, options.end_pos) {
            (Some(start), Some(end)) if start < end && end <= char_len => {
                Ok(Placement::Select { start, end })
            }
            (Some(start), Some(end)) => Err(Error::validation(format!(
                "select range {start}..{end} is invalid for a prompt of {char_len} \
                 characters; requires 0 <= start_pos < end_pos <= {char_len}"
            ))),
            _ => Err(Error::validation(
                "select mode requires both start_pos and end_pos, satisfying \
                 0 <= start_pos < end_pos <= len",
            )),
        },
    }
}

/// Runs the intent call and returns the feedback phrasing the optimize call uses.
///
/// The intent template sees the **original, untagged** prompt. When the answer
/// parses and both the intent flag and the phrasing are usable, the trimmed
/// phrasing is returned; otherwise the raw [`FeedbackOptions::feedback`] is
/// returned verbatim after a `debug` log.
async fn suggestion(model: &dyn Model, prompt: &str, options: &FeedbackOptions) -> Result<String> {
    let messages = render(
        bank(options.language).feedback_intent,
        &[
            ("original_prompt", prompt),
            ("feedbacks", options.feedback.as_str()),
        ],
    );
    let request = ModelRequestConfig::new(model.model_name());
    let response = model.invoke(&messages, &request).await?;
    let answer = response
        .message()
        .map(|message| message.text().into_owned());

    let normalized = answer
        .as_deref()
        .and_then(parse_feedback_intent)
        .filter(|intent| {
            intent
                .intent
                .as_deref()
                .is_some_and(|flag| flag.eq_ignore_ascii_case("true"))
        })
        .and_then(|intent| intent.optimized_feedback)
        .filter(|text| !text.trim().is_empty());

    if let Some(text) = normalized {
        return Ok(text.trim().to_owned());
    }

    tracing::debug!(
        model = model.model_name(),
        "feedback intent was unusable; using the raw feedback"
    );
    Ok(options.feedback.clone())
}

/// Assembles the optimize messages shared by `build` and `stream_build`.
///
/// Validates first, then makes the intent call, then renders the mode template.
async fn optimize_messages(
    model: &dyn Model,
    prompt: &str,
    options: &FeedbackOptions,
) -> Result<Vec<BaseMessage>> {
    let placement = validate(prompt, options)?;
    let suggestion = suggestion(model, prompt, options).await?;
    let prompt_bank = bank(options.language);

    let messages = match placement {
        Placement::General => render(
            prompt_bank.feedback_general,
            &[
                ("original_prompt", prompt),
                ("suggestion", suggestion.as_str()),
            ],
        ),
        Placement::Insert(start) => {
            let tagged = insert_marker(prompt, start)?;
            render(
                prompt_bank.feedback_insert,
                &[
                    ("original_prompt", tagged.as_str()),
                    ("suggestion", suggestion.as_str()),
                ],
            )
        }
        Placement::Select { start, end } => {
            let pending = select_range(prompt, start, end)?;
            render(
                prompt_bank.feedback_select,
                &[
                    ("original_prompt", prompt),
                    ("pending_optimized_prompt", pending.as_str()),
                    ("suggestion", suggestion.as_str()),
                ],
            )
        }
    };
    Ok(messages)
}

/// Renders one single-message user template with the given `values`.
///
/// Shared by both entry points and the tests, so a rendered conversation is
/// always assembled the same way.
fn render(template: &str, values: &[(&str, &str)]) -> Vec<BaseMessage> {
    let values: BTreeMap<String, Value> = values
        .iter()
        .map(|(key, value)| ((*key).to_owned(), Value::String((*value).to_owned())))
        .collect();
    user_template(template).format(&values).to_messages()
}

/// Extracts the first choice's text, rejecting an unusable answer.
fn answer_text(response: &ChatResponse, model_name: &str) -> Result<String> {
    let Some(message) = response.message() else {
        return Err(Error::llm(format!(
            "model {model_name:?} returned no completion choice"
        )));
    };
    let text = message.text().into_owned();
    if text.trim().is_empty() {
        return Err(Error::llm(format!(
            "model {model_name:?} returned an empty answer"
        )));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_test_support::{CallMode, ScriptedModel, ScriptedTurn};

    use super::*;
    use crate::prompt_builder::INSERT_STR;

    /// An intent answer that accepts the feedback and normalizes it.
    const INTENT_JSON: &str =
        "```json\n{\"intent\": \"true\", \"optimized_feedback\": \"be concise\"}\n```";

    /// Builds a recording model from `turns` and the `Arc<dyn Model>` view.
    fn scripted(turns: impl IntoIterator<Item = ScriptedTurn>) -> (ScriptedModel, Arc<dyn Model>) {
        let recorder = ScriptedModel::new(turns);
        let model: Arc<dyn Model> = Arc::new(recorder.clone());
        (recorder, model)
    }

    /// An accepted intent turn followed by `answer`.
    fn scripted_two(answer: &str) -> (ScriptedModel, Arc<dyn Model>) {
        scripted([ScriptedTurn::text(INTENT_JSON), ScriptedTurn::text(answer)])
    }

    /// Default options carrying `feedback`.
    fn options(feedback: &str) -> FeedbackOptions {
        FeedbackOptions {
            feedback: feedback.to_owned(),
            ..FeedbackOptions::default()
        }
    }

    /// `template` rendered with `values` through the same single-message path.
    fn expected(template: &str, values: &[(&str, &str)]) -> String {
        render(template, values)[0].text().into_owned()
    }

    /// The text of one recorded call's only message.
    fn call_text(recorder: &ScriptedModel, index: usize) -> String {
        recorder.calls()[index].messages[0].text().into_owned()
    }

    #[tokio::test]
    async fn general_build_makes_two_calls_and_returns_the_answer() {
        let (recorder, model) = scripted_two("optimized");
        let builder = FeedbackPromptBuilder::new(model);

        let answer = builder
            .build("write a poem", &options("make it shorter"))
            .await
            .unwrap();

        assert_eq!(answer, "optimized");
        assert_eq!(recorder.call_count(), 2);
        let calls = recorder.calls();
        assert_eq!(calls[0].mode, CallMode::Invoke);
        assert_eq!(calls[1].mode, CallMode::Invoke);

        let intent_text = call_text(&recorder, 0);
        let wanted_intent = expected(
            bank(Language::ZhCn).feedback_intent,
            &[
                ("original_prompt", "write a poem"),
                ("feedbacks", "make it shorter"),
            ],
        );
        assert_eq!(intent_text, wanted_intent);
        assert!(intent_text.contains("write a poem"));
        assert!(intent_text.contains("make it shorter"));
        assert!(
            !intent_text.contains("{{"),
            "unrendered slot: {intent_text}"
        );

        let optimize_text = call_text(&recorder, 1);
        let wanted_optimize = expected(
            bank(Language::ZhCn).feedback_general,
            &[
                ("original_prompt", "write a poem"),
                ("suggestion", "be concise"),
            ],
        );
        assert_eq!(optimize_text, wanted_optimize);
        assert!(optimize_text.contains("be concise"));
        assert!(!optimize_text.contains("make it shorter"));
        assert!(!optimize_text.contains("{{"));
    }

    #[tokio::test]
    async fn intent_answer_without_json_falls_back_to_the_raw_feedback() {
        let (recorder, model) =
            scripted([ScriptedTurn::text("sorry"), ScriptedTurn::text("optimized")]);
        let builder = FeedbackPromptBuilder::new(model);

        let answer = builder
            .build("write a poem", &options("make it shorter"))
            .await
            .unwrap();

        assert_eq!(answer, "optimized");
        let optimize_text = call_text(&recorder, 1);
        let wanted = expected(
            bank(Language::ZhCn).feedback_general,
            &[
                ("original_prompt", "write a poem"),
                ("suggestion", "make it shorter"),
            ],
        );
        assert_eq!(optimize_text, wanted);
    }

    #[tokio::test]
    async fn a_rejected_intent_falls_back_to_the_raw_feedback() {
        let rejected = "```json\n{\"intent\": \"false\", \"optimized_feedback\": \"x\"}\n```";
        let (recorder, model) = scripted([
            ScriptedTurn::text(rejected),
            ScriptedTurn::text("optimized"),
        ]);
        let builder = FeedbackPromptBuilder::new(model);

        builder
            .build("write a poem", &options("make it shorter"))
            .await
            .unwrap();

        let optimize_text = call_text(&recorder, 1);
        let wanted = expected(
            bank(Language::ZhCn).feedback_general,
            &[
                ("original_prompt", "write a poem"),
                ("suggestion", "make it shorter"),
            ],
        );
        assert_eq!(optimize_text, wanted);
        assert!(!optimize_text.contains("be concise"));
    }

    #[tokio::test]
    async fn insert_splices_the_marker_into_the_optimize_prompt_only() {
        let (recorder, model) = scripted_two("inserted");
        let builder = FeedbackPromptBuilder::new(model);
        let options = FeedbackOptions {
            feedback: "add a comma".to_owned(),
            mode: FeedbackMode::Insert,
            start_pos: Some(3),
            ..FeedbackOptions::default()
        };

        let answer = builder.build("abcdef", &options).await.unwrap();
        assert_eq!(answer, "inserted");

        let intent_text = call_text(&recorder, 0);
        assert!(intent_text.contains("abcdef"), "{intent_text}");
        assert!(!intent_text.contains(INSERT_STR));

        let optimize_text = call_text(&recorder, 1);
        assert!(optimize_text.contains("abc[用户要插入的位置]def"));
        let tagged = format!("abc{INSERT_STR}def");
        let wanted = expected(
            bank(Language::ZhCn).feedback_insert,
            &[("original_prompt", &tagged), ("suggestion", "be concise")],
        );
        assert_eq!(optimize_text, wanted);
    }

    #[tokio::test]
    async fn insert_at_the_character_length_appends_the_marker() {
        let (recorder, model) = scripted_two("inserted");
        let builder = FeedbackPromptBuilder::new(model);
        let options = FeedbackOptions {
            feedback: "add a suffix".to_owned(),
            mode: FeedbackMode::Insert,
            start_pos: Some(6),
            ..FeedbackOptions::default()
        };

        builder.build("abcdef", &options).await.unwrap();

        let optimize_text = call_text(&recorder, 1);
        assert!(optimize_text.contains(&format!("abcdef{INSERT_STR}")));
    }

    #[tokio::test]
    async fn insert_positions_are_character_indices() {
        let (recorder, model) = scripted_two("inserted");
        let builder = FeedbackPromptBuilder::new(model);
        let options = FeedbackOptions {
            feedback: "add a word".to_owned(),
            mode: FeedbackMode::Insert,
            start_pos: Some(2),
            ..FeedbackOptions::default()
        };

        builder.build("你好世界", &options).await.unwrap();

        let optimize_text = call_text(&recorder, 1);
        assert!(
            optimize_text.contains(&format!("你好{INSERT_STR}世界")),
            "{optimize_text}"
        );
    }

    #[tokio::test]
    async fn select_keeps_the_full_prompt_and_the_pending_fragment() {
        let (recorder, model) = scripted_two("rewritten");
        let builder = FeedbackPromptBuilder::new(model);
        let options = FeedbackOptions {
            feedback: "make it formal".to_owned(),
            mode: FeedbackMode::Select,
            start_pos: Some(1),
            end_pos: Some(4),
            ..FeedbackOptions::default()
        };

        let answer = builder.build("abcdef", &options).await.unwrap();
        assert_eq!(answer, "rewritten");

        let optimize_text = call_text(&recorder, 1);
        let wanted = expected(
            bank(Language::ZhCn).feedback_select,
            &[
                ("original_prompt", "abcdef"),
                ("pending_optimized_prompt", "bcd"),
                ("suggestion", "be concise"),
            ],
        );
        assert_eq!(optimize_text, wanted);
        assert!(optimize_text.contains("abcdef"));
        assert!(optimize_text.contains("bcd"));
    }

    #[tokio::test]
    async fn select_ranges_are_character_indices() {
        let (recorder, model) = scripted_two("rewritten");
        let builder = FeedbackPromptBuilder::new(model);
        let options = FeedbackOptions {
            feedback: "make it formal".to_owned(),
            mode: FeedbackMode::Select,
            start_pos: Some(1),
            end_pos: Some(3),
            ..FeedbackOptions::default()
        };

        builder.build("你好世界", &options).await.unwrap();

        let optimize_text = call_text(&recorder, 1);
        assert!(optimize_text.contains("好世"), "{optimize_text}");
        assert!(optimize_text.contains("你好世界"), "{optimize_text}");
    }

    #[tokio::test]
    async fn general_mode_ignores_the_positions() {
        let (recorder, model) = scripted_two("optimized");
        let builder = FeedbackPromptBuilder::new(model);
        let options = FeedbackOptions {
            feedback: "make it shorter".to_owned(),
            mode: FeedbackMode::General,
            start_pos: Some(4),
            end_pos: Some(2),
            ..FeedbackOptions::default()
        };

        builder.build("abcdef", &options).await.unwrap();

        let optimize_text = call_text(&recorder, 1);
        assert!(!optimize_text.contains(INSERT_STR));
        let wanted = expected(
            bank(Language::ZhCn).feedback_general,
            &[("original_prompt", "abcdef"), ("suggestion", "be concise")],
        );
        assert_eq!(optimize_text, wanted);
    }

    /// Runs `build` with `prompt` and `options`, asserting a validation error
    /// and that no model turn was consumed.
    async fn assert_validation_error(
        recorder: &ScriptedModel,
        prompt: &str,
        options: &FeedbackOptions,
    ) {
        let model: Arc<dyn Model> = Arc::new(recorder.clone());
        let builder = FeedbackPromptBuilder::new(model);

        let error = builder.build(prompt, options).await.unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Validation, "{error}");
        assert_eq!(recorder.call_count(), 0);
    }

    #[tokio::test]
    async fn a_blank_prompt_is_a_validation_error() {
        let (recorder, _model) = scripted_two("optimized");
        assert_validation_error(&recorder, "   ", &options("make it shorter")).await;
    }

    #[tokio::test]
    async fn a_blank_feedback_is_a_validation_error() {
        let (recorder, _model) = scripted_two("optimized");
        assert_validation_error(&recorder, "abcdef", &options("  \n")).await;
    }

    #[tokio::test]
    async fn insert_without_a_position_is_a_validation_error() {
        let (recorder, _model) = scripted_two("optimized");
        let options = FeedbackOptions {
            feedback: "add a comma".to_owned(),
            mode: FeedbackMode::Insert,
            ..FeedbackOptions::default()
        };
        assert_validation_error(&recorder, "abcdef", &options).await;
    }

    #[tokio::test]
    async fn insert_past_the_end_is_a_validation_error() {
        let (recorder, _model) = scripted_two("optimized");
        let options = FeedbackOptions {
            feedback: "add a comma".to_owned(),
            mode: FeedbackMode::Insert,
            start_pos: Some(7),
            ..FeedbackOptions::default()
        };
        assert_validation_error(&recorder, "abcdef", &options).await;
    }

    #[tokio::test]
    async fn select_without_a_start_position_is_a_validation_error() {
        let (recorder, _model) = scripted_two("optimized");
        let options = FeedbackOptions {
            feedback: "rewrite it".to_owned(),
            mode: FeedbackMode::Select,
            end_pos: Some(4),
            ..FeedbackOptions::default()
        };
        assert_validation_error(&recorder, "abcdef", &options).await;
    }

    #[tokio::test]
    async fn select_without_an_end_position_is_a_validation_error() {
        let (recorder, _model) = scripted_two("optimized");
        let options = FeedbackOptions {
            feedback: "rewrite it".to_owned(),
            mode: FeedbackMode::Select,
            start_pos: Some(1),
            ..FeedbackOptions::default()
        };
        assert_validation_error(&recorder, "abcdef", &options).await;
    }

    #[tokio::test]
    async fn select_with_an_inverted_range_is_a_validation_error() {
        let (recorder, _model) = scripted_two("optimized");
        let options = FeedbackOptions {
            feedback: "rewrite it".to_owned(),
            mode: FeedbackMode::Select,
            start_pos: Some(4),
            end_pos: Some(4),
            ..FeedbackOptions::default()
        };
        assert_validation_error(&recorder, "abcdef", &options).await;
    }

    #[tokio::test]
    async fn select_past_the_end_is_a_validation_error() {
        let (recorder, _model) = scripted_two("optimized");
        let options = FeedbackOptions {
            feedback: "rewrite it".to_owned(),
            mode: FeedbackMode::Select,
            start_pos: Some(1),
            end_pos: Some(7),
            ..FeedbackOptions::default()
        };
        assert_validation_error(&recorder, "abcdef", &options).await;
    }

    #[tokio::test]
    async fn an_empty_optimize_answer_is_an_llm_error() {
        let (recorder, model) = scripted_two("");
        let builder = FeedbackPromptBuilder::new(model);

        let error = builder
            .build("abcdef", &options("make it shorter"))
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Llm);
        assert_eq!(recorder.call_count(), 2);
    }

    #[test]
    fn feedback_mode_serde_uses_the_python_tags() {
        for (mode, tag) in [
            (FeedbackMode::General, "\"general\""),
            (FeedbackMode::Insert, "\"insert\""),
            (FeedbackMode::Select, "\"select\""),
        ] {
            assert_eq!(serde_json::to_string(&mode).unwrap(), tag);
            let parsed: FeedbackMode = serde_json::from_str(tag).unwrap();
            assert_eq!(parsed, mode);
        }
        assert_eq!(FeedbackMode::default(), FeedbackMode::General);
    }

    #[test]
    fn feedback_options_serde_round_trips_and_tolerates_unknown_fields() {
        let json = r#"{"feedback":"be brief","mode":"select","start_pos":1,"end_pos":3,"language":"en-US","unknown":true}"#;
        let options: FeedbackOptions = serde_json::from_str(json).unwrap();
        assert_eq!(options.feedback, "be brief");
        assert_eq!(options.mode, FeedbackMode::Select);
        assert_eq!(options.start_pos, Some(1));
        assert_eq!(options.end_pos, Some(3));
        assert_eq!(options.language, Language::EnUs);

        let value = serde_json::to_value(&options).unwrap();
        let object = value.as_object().unwrap();
        for key in ["feedback", "mode", "start_pos", "end_pos", "language"] {
            assert!(object.contains_key(key), "missing field {key}");
        }
        assert_eq!(object["mode"], Value::String("select".to_owned()));
        assert_eq!(object["language"], Value::String("en-US".to_owned()));

        let restored: FeedbackOptions = serde_json::from_value(value).unwrap();
        assert_eq!(restored, options);

        let minimal: FeedbackOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(minimal, FeedbackOptions::default());
    }

    #[test]
    fn default_options_match_python() {
        let options = FeedbackOptions::default();
        assert_eq!(options.feedback, "");
        assert_eq!(options.mode, FeedbackMode::General);
        assert!(options.start_pos.is_none());
        assert!(options.end_pos.is_none());
        assert_eq!(options.language, Language::ZhCn);
    }

    #[tokio::test]
    async fn stream_build_forwards_the_optimize_answer() {
        let (recorder, model) = scripted_two("streamed answer");
        let builder = FeedbackPromptBuilder::new(model);

        let items: Vec<Result<String>> = builder
            .stream_build("write a poem".to_owned(), options("make it shorter"))
            .collect()
            .await;

        assert!(items.iter().all(std::result::Result::is_ok), "{items:?}");
        let text: String = items.into_iter().map(|item| item.unwrap()).collect();
        assert_eq!(text, "streamed answer");

        assert_eq!(recorder.call_count(), 2);
        let calls = recorder.calls();
        assert_eq!(calls[0].mode, CallMode::Invoke);
        assert_eq!(calls[1].mode, CallMode::Stream);
    }

    #[tokio::test]
    async fn stream_build_yields_one_validation_error_item() {
        let (recorder, model) = scripted_two("unused");
        let builder = FeedbackPromptBuilder::new(model);

        let items: Vec<Result<String>> = builder
            .stream_build("   ".to_owned(), options("make it shorter"))
            .collect()
            .await;

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].as_ref().unwrap_err().kind(), ErrorKind::Validation);
        assert_eq!(recorder.call_count(), 0);
    }

    #[tokio::test]
    async fn stream_build_yields_a_model_error_and_stops() {
        let (recorder, model) =
            scripted([ScriptedTurn::text(INTENT_JSON), ScriptedTurn::fail("boom")]);
        let builder = FeedbackPromptBuilder::new(model);

        let items: Vec<Result<String>> = builder
            .stream_build("abcdef".to_owned(), options("make it shorter"))
            .collect()
            .await;

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].as_ref().unwrap_err().kind(), ErrorKind::Llm);
        assert_eq!(recorder.call_count(), 2);
    }
}
