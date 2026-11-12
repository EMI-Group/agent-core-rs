//! Bad-case prompt repair.
//!
//! Ports `openjiuwen.dev_tools.prompt_builder.builder.badcase_prompt_builder`
//! (`BadCasePromptBuilder`). The builder repairs a prompt from its failures:
//! evaluated cases are rendered into the bank's case format, one **analyze**
//! model call condenses them into a feedback summary, and one **optimize** call
//! rewrites the prompt from that summary. Exactly two model calls are made, the
//! same sequence as Python's `_get_feedback_from_bad_case` followed by
//! `_format_bad_case_template`.
//!
//! # Deviations from Python
//!
//! - Validation is strict and happens before any model call: a blank prompt, an
//!   empty case list, and more than [`MAX_CASES_LIMIT`] cases are
//!   [`Error::Validation`] instead of Python's `build_error` raised mid-flight.
//! - A model answer with no choice or with blank text is [`Error::Llm`]; Python
//!   returns `response.content`, which may be `None` or empty.
//! - A rejected `<intent>false</intent>` only logs at `debug` level and the
//!   repair proceeds with the summary, matching Python's warn-and-continue.
//! - [`BadCasePromptBuilder::stream_build`] runs the analyze stage as a
//!   non-streamed `invoke` (it is a prerequisite, not model-facing output), so
//!   the stream always carries only the final repaired prompt.
//! - Cases render their maps as compact JSON instead of Python's `str(dict)`
//!   repr (see [`super::utils::render_bad_cases`]).

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{ChatResponse, Model, ModelRequestConfig, StreamChunk};
use openjiuwen_foundation::prompt::BaseMessage;
use serde_json::Value;

use super::language::Language;
use super::templates::{bank, user_template, PromptBank};
use super::utils::{parse_bad_case_analysis, render_bad_cases};
use crate::tune::EvaluatedCase;

/// Maximum number of evaluated cases one repair call accepts.
///
/// Ports Python's `MAX_CASES_LIMIT`. A larger list is a validation error rather
/// than a truncated request, so the caller never silently loses cases.
pub const MAX_CASES_LIMIT: usize = 10;

/// Repairs a prompt from its evaluated failures with two model calls.
///
/// The first call analyzes the rendered cases and yields a feedback summary; the
/// second call rewrites the original prompt from that summary. The builder owns
/// no configuration: the caller supplies the configured model and keeps
/// ownership of its request settings.
pub struct BadCasePromptBuilder {
    /// Model used for the analyze and optimize calls.
    model: Arc<dyn Model>,
}

impl BadCasePromptBuilder {
    /// Creates a builder over `model`.
    ///
    /// The model is shared, never rebuilt, so the caller keeps ownership of its
    /// configuration.
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self { model }
    }

    /// Analyzes `cases` and returns the repaired prompt.
    ///
    /// Exactly two model calls are made: the analyze template receives the
    /// original prompt plus the rendered cases, and the optimize template
    /// receives the original prompt plus the parsed feedback summary.
    ///
    /// # Errors
    ///
    /// - `Error::Validation` when the prompt is blank, `cases` is empty, or
    ///   `cases` holds more than [`MAX_CASES_LIMIT`] entries.
    /// - `Error::Llm` when a model answer carries no choice or blank text.
    /// - Any error the model returns, unchanged.
    pub async fn build(
        &self,
        prompt: &str,
        cases: &[EvaluatedCase],
        language: Language,
    ) -> Result<String> {
        validate(prompt, cases)?;
        let prompt_bank = bank(language);
        let analyze = analyze_messages(prompt, cases, prompt_bank)?;
        let request = ModelRequestConfig::new(self.model.model_name());

        let response = self.model.invoke(&analyze, &request).await?;
        let analysis = answer_text(&response, self.model.model_name())?;
        let feedback = feedback_summary(&analysis);

        let optimize = optimize_messages(prompt, &feedback, prompt_bank);
        let response = self.model.invoke(&optimize, &request).await?;
        answer_text(&response, self.model.model_name())
    }

    /// Analyzes `cases` once and streams the repaired prompt.
    ///
    /// The returned stream owns a clone of the [`Arc<dyn Model>`] and of its
    /// inputs, so the builder is not borrowed and may be dropped. Because the
    /// analyze stage is a non-streamed `invoke`, the stream carries only the
    /// final repaired prompt: only non-empty text deltas of the optimize call
    /// are forwarded as `Ok(String)`, while usage, done, and empty deltas are
    /// skipped.
    ///
    /// A validation, rendering, or analyze error is yielded as the single `Err`
    /// item and ends the stream; a model error during streaming is yielded as
    /// `Err` before the stream terminates.
    pub fn stream_build(
        &self,
        prompt: String,
        cases: Vec<EvaluatedCase>,
        language: Language,
    ) -> BoxStream<'static, Result<String>> {
        let model = Arc::clone(&self.model);

        Box::pin(async_stream::stream! {
            if let Err(error) = validate(&prompt, &cases) {
                yield Err(error);
                return;
            }
            let prompt_bank = bank(language);
            let analyze = match analyze_messages(&prompt, &cases, prompt_bank) {
                Ok(messages) => messages,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            let request = ModelRequestConfig::new(model.model_name());
            let response = match model.invoke(&analyze, &request).await {
                Ok(response) => response,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            let analysis = match answer_text(&response, model.model_name()) {
                Ok(text) => text,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            let feedback = feedback_summary(&analysis);

            let optimize = optimize_messages(&prompt, &feedback, prompt_bank);
            let mut stream = model.stream(optimize, ModelRequestConfig::new(model.model_name()));
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

/// Checks the Python `_validate_input` constraints before any model call.
///
/// # Errors
///
/// [`Error::Validation`] naming the violated constraint.
fn validate(prompt: &str, cases: &[EvaluatedCase]) -> Result<()> {
    if prompt.trim().is_empty() {
        return Err(Error::validation("prompt cannot be empty"));
    }
    if cases.is_empty() {
        return Err(Error::validation("the cases cannot be empty"));
    }
    if cases.len() > MAX_CASES_LIMIT {
        return Err(Error::validation(format!(
            "the number of cases cannot exceed {MAX_CASES_LIMIT}"
        )));
    }
    Ok(())
}

/// Builds the analyze-call messages shared by `build` and `stream_build`.
///
/// # Errors
///
/// [`Error::Validation`] when the cases cannot be rendered (the bank's case
/// format is plain text, so this never happens in practice).
fn analyze_messages(
    prompt: &str,
    cases: &[EvaluatedCase],
    prompt_bank: &PromptBank,
) -> Result<Vec<BaseMessage>> {
    let bad_cases = render_bad_cases(cases, prompt_bank)?;
    let values = value_pairs([("original_prompt", prompt), ("bad_cases", &bad_cases)]);
    Ok(user_template(prompt_bank.bad_case_analyze)
        .format(&values)
        .to_messages())
}

/// Builds the optimize-call messages from the original prompt and the summary.
fn optimize_messages(prompt: &str, feedback: &str, prompt_bank: &PromptBank) -> Vec<BaseMessage> {
    let values = value_pairs([("original_prompt", prompt), ("feedback", feedback)]);
    user_template(prompt_bank.bad_case_optimize)
        .format(&values)
        .to_messages()
}

/// Turns `(name, text)` pairs into the value map [`super::templates::user_template`]
/// expects.
fn value_pairs(entries: [(&str, &str); 2]) -> BTreeMap<String, Value> {
    entries
        .into_iter()
        .map(|(key, value)| (key.to_owned(), Value::String(value.to_owned())))
        .collect()
}

/// Parses the analysis answer and logs a rejected intent.
///
/// Mirrors Python's `_parse_feedback_summary`: the summary is used either way,
/// and only an explicit `false` intent is noticed.
fn feedback_summary(analysis: &str) -> String {
    let (intent, summary) = parse_bad_case_analysis(analysis);
    if !intent {
        tracing::debug!(
            summary = %summary,
            "bad-case analysis did not confirm the failures; repairing the prompt anyway"
        );
    }
    summary
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
    use futures::StreamExt;
    use openjiuwen_common::error::{ErrorKind, Result};
    use openjiuwen_foundation::prompt::Role;
    use openjiuwen_test_support::{CallMode, ScriptedModel, ScriptedTurn};

    use super::*;
    use crate::prompt_builder::templates::{self, bank};
    use crate::tune::Case;

    /// A well-formed analysis answer confirming the intent.
    const ANALYZE_ANSWER: &str = "<intent>true</intent>\n<summary>be clearer</summary>";

    /// A map from `(key, value)` pairs.
    fn map(entries: &[(&str, Value)]) -> BTreeMap<String, Value> {
        entries
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.clone()))
            .collect()
    }

    /// An evaluated case with a fixed id and a neutral score.
    fn evaluated(id: &str, reason: &str) -> EvaluatedCase {
        let case = Case::new(
            map(&[("question", Value::String(format!("question-{id}")))]),
            map(&[("answer", Value::String("expected".to_owned()))]),
        )
        .unwrap()
        .with_case_id(id);
        EvaluatedCase::new(
            case,
            map(&[("answer", Value::String("actual".to_owned()))]),
            0.25,
            reason,
        )
        .unwrap()
    }

    /// Builds a recording model and the `Arc<dyn Model>` view handed to the
    /// builder; both share one script and call history.
    fn scripted(turns: Vec<ScriptedTurn>) -> (ScriptedModel, Arc<dyn Model>) {
        let recorder = ScriptedModel::new(turns);
        let model: Arc<dyn Model> = Arc::new(recorder.clone());
        (recorder, model)
    }

    /// A model scripted with one analyze turn and one optimize turn.
    fn two_turn(analyze: &str, optimize: &str) -> (ScriptedModel, Arc<dyn Model>) {
        scripted(vec![
            ScriptedTurn::text(analyze),
            ScriptedTurn::text(optimize),
        ])
    }

    /// The text a template renders to, for message assertions.
    fn rendered(template: &str, values: &BTreeMap<String, Value>) -> String {
        user_template(template).format(values).to_messages()[0]
            .text()
            .into_owned()
    }

    /// The value map the optimize template receives.
    fn optimize_values(prompt: &str, feedback: &str) -> BTreeMap<String, Value> {
        map(&[
            ("original_prompt", Value::String(prompt.to_owned())),
            ("feedback", Value::String(feedback.to_owned())),
        ])
    }

    #[tokio::test]
    async fn build_makes_two_calls_and_returns_the_optimized_prompt() {
        let (recorder, model) = two_turn(ANALYZE_ANSWER, "repaired");
        let builder = BadCasePromptBuilder::new(model);
        let cases = vec![evaluated("c1", "too vague")];

        let answer = builder
            .build("original prompt", &cases, Language::ZhCn)
            .await
            .unwrap();

        assert_eq!(answer, "repaired");
        assert_eq!(recorder.call_count(), 2);
        let zh = bank(Language::ZhCn);
        let calls = recorder.calls();

        // Call 1: the analyze template, with the original prompt and the
        // rendered case record.
        let bad_cases = render_bad_cases(&cases, zh).unwrap();
        let analyze_text = calls[0].messages[0].text().into_owned();
        assert_eq!(calls[0].messages.len(), 1);
        assert_eq!(calls[0].messages[0].role, Role::User);
        assert_eq!(
            analyze_text,
            rendered(
                zh.bad_case_analyze,
                &map(&[
                    (
                        "original_prompt",
                        Value::String("original prompt".to_owned())
                    ),
                    ("bad_cases", Value::String(bad_cases)),
                ])
            )
        );
        assert!(analyze_text.contains("original prompt"));
        assert!(analyze_text.contains("[question]"));
        assert!(analyze_text.contains("\"question\":\"question-c1\""));
        assert!(analyze_text.contains("too vague"));
        assert!(!analyze_text.contains("{{"));

        // Call 2: the optimize template, with the parsed summary as feedback.
        assert_eq!(
            calls[1].messages[0].text(),
            rendered(
                zh.bad_case_optimize,
                &optimize_values("original prompt", "be clearer")
            )
        );
        assert_eq!(calls[0].request.model_name, recorder.model_name());
    }

    #[tokio::test]
    async fn a_missing_summary_becomes_the_trimmed_analysis() {
        let (recorder, model) = two_turn("  plain analysis  ", "repaired");
        let builder = BadCasePromptBuilder::new(model);

        builder
            .build(
                "original prompt",
                &[evaluated("c1", "too vague")],
                Language::ZhCn,
            )
            .await
            .unwrap();

        let calls = recorder.calls();
        assert_eq!(
            calls[1].messages[0].text(),
            rendered(
                bank(Language::ZhCn).bad_case_optimize,
                &optimize_values("original prompt", "plain analysis")
            )
        );
    }

    #[tokio::test]
    async fn a_rejected_intent_still_runs_the_optimize_call() {
        let (recorder, model) = two_turn(
            "<intent>false</intent>\n<summary>still repair</summary>",
            "repaired",
        );
        let builder = BadCasePromptBuilder::new(model);

        let answer = builder
            .build(
                "original prompt",
                &[evaluated("c1", "too vague")],
                Language::ZhCn,
            )
            .await
            .unwrap();

        assert_eq!(answer, "repaired");
        assert_eq!(recorder.call_count(), 2);
        assert!(recorder.calls()[1].messages[0]
            .text()
            .contains("still repair"));
    }

    #[tokio::test]
    async fn multiple_cases_are_joined_with_the_bank_separator() {
        let (recorder, model) = two_turn(ANALYZE_ANSWER, "repaired");
        let builder = BadCasePromptBuilder::new(model);
        let cases = vec![
            evaluated("c1", "first reason"),
            evaluated("c2", "second reason"),
        ];

        builder
            .build("original prompt", &cases, Language::ZhCn)
            .await
            .unwrap();

        let zh = bank(Language::ZhCn);
        let rendered_cases = render_bad_cases(&cases, zh).unwrap();
        // Two records, each ending with the `=== ` separator and joined by one
        // newline, so consecutive records are separated by one blank line.
        assert!(rendered_cases.contains("=== "));
        assert!(rendered_cases.contains("\n\n"));

        let analyze_text = recorder.calls()[0].messages[0].text().into_owned();
        assert!(analyze_text.contains("=== "));
        assert!(analyze_text.contains("first reason"));
        assert!(analyze_text.contains("second reason"));
        assert_eq!(
            analyze_text.matches("first reason").count(),
            1,
            "each reason appears exactly once"
        );
    }

    #[tokio::test]
    async fn english_language_uses_the_en_bank() {
        let (recorder, model) = two_turn(ANALYZE_ANSWER, "repaired");
        let builder = BadCasePromptBuilder::new(model);
        let cases = vec![evaluated("c1", "too vague")];

        builder
            .build("original prompt", &cases, Language::EnUs)
            .await
            .unwrap();

        let en = &templates::en_us::BANK;
        let zh = bank(Language::ZhCn);
        let analyze_text = rendered(
            en.bad_case_analyze,
            &map(&[
                (
                    "original_prompt",
                    Value::String("original prompt".to_owned()),
                ),
                (
                    "bad_cases",
                    Value::String(render_bad_cases(&cases, en).unwrap()),
                ),
            ]),
        );
        assert_eq!(recorder.calls()[0].messages[0].text(), analyze_text);
        assert_ne!(
            analyze_text,
            rendered(
                zh.bad_case_analyze,
                &map(&[
                    (
                        "original_prompt",
                        Value::String("original prompt".to_owned())
                    ),
                    (
                        "bad_cases",
                        Value::String(render_bad_cases(&cases, zh).unwrap())
                    ),
                ])
            )
        );
    }

    #[tokio::test]
    async fn invalid_input_is_rejected_without_a_model_call() {
        let (recorder, model) = two_turn(ANALYZE_ANSWER, "repaired");
        let builder = BadCasePromptBuilder::new(model);
        let one = vec![evaluated("c1", "too vague")];

        for prompt in ["", "  ", "\n\t"] {
            let error = builder
                .build(prompt, &one, Language::ZhCn)
                .await
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "{prompt:?}");
            assert!(error.to_string().contains("prompt"));
        }

        let error = builder
            .build("original prompt", &[], Language::ZhCn)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("cases"));

        let too_many: Vec<EvaluatedCase> = (0..=MAX_CASES_LIMIT)
            .map(|index| evaluated(&format!("c{index}"), "too vague"))
            .collect();
        assert_eq!(too_many.len(), MAX_CASES_LIMIT + 1);
        let error = builder
            .build("original prompt", &too_many, Language::ZhCn)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains(&MAX_CASES_LIMIT.to_string()));

        assert_eq!(recorder.call_count(), 0);
        assert_eq!(MAX_CASES_LIMIT, 10);
    }

    #[tokio::test]
    async fn exactly_the_limit_is_accepted() {
        let (recorder, model) = two_turn(ANALYZE_ANSWER, "repaired");
        let builder = BadCasePromptBuilder::new(model);
        let cases: Vec<EvaluatedCase> = (0..MAX_CASES_LIMIT)
            .map(|index| evaluated(&format!("c{index}"), "too vague"))
            .collect();

        let answer = builder
            .build("original prompt", &cases, Language::ZhCn)
            .await
            .unwrap();

        assert_eq!(answer, "repaired");
        assert_eq!(recorder.call_count(), 2);
    }

    #[tokio::test]
    async fn an_empty_optimized_answer_is_an_llm_error() {
        let (recorder, model) = scripted(vec![
            ScriptedTurn::text(ANALYZE_ANSWER),
            ScriptedTurn::text("   "),
        ]);
        let builder = BadCasePromptBuilder::new(model);

        let error = builder
            .build(
                "original prompt",
                &[evaluated("c1", "too vague")],
                Language::ZhCn,
            )
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Llm);
        assert_eq!(recorder.call_count(), 2);
    }

    #[tokio::test]
    async fn stream_build_analyzes_then_streams_the_optimization() {
        let (recorder, model) = two_turn(ANALYZE_ANSWER, "repaired answer");
        let builder = BadCasePromptBuilder::new(model);
        let cases = vec![evaluated("c1", "too vague")];

        let items: Vec<Result<String>> = builder
            .stream_build("original prompt".to_owned(), cases, Language::ZhCn)
            .collect()
            .await;

        assert!(items.iter().all(std::result::Result::is_ok), "{items:?}");
        let text: String = items.into_iter().map(|item| item.unwrap()).collect();
        assert_eq!(text, "repaired answer");

        let calls = recorder.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].mode, CallMode::Invoke);
        assert_eq!(calls[1].mode, CallMode::Stream);
        assert_eq!(
            calls[1].messages[0].text(),
            rendered(
                bank(Language::ZhCn).bad_case_optimize,
                &optimize_values("original prompt", "be clearer")
            )
        );
    }

    #[tokio::test]
    async fn stream_build_yields_one_validation_error() {
        let (recorder, model) = two_turn(ANALYZE_ANSWER, "repaired");
        let builder = BadCasePromptBuilder::new(model);

        let items: Vec<Result<String>> = builder
            .stream_build(
                "   ".to_owned(),
                vec![evaluated("c1", "too vague")],
                Language::ZhCn,
            )
            .collect()
            .await;

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].as_ref().unwrap_err().kind(), ErrorKind::Validation);
        assert_eq!(recorder.call_count(), 0);
    }

    #[tokio::test]
    async fn stream_build_yields_a_failing_analysis_as_one_error() {
        let (recorder, model) = scripted(vec![
            ScriptedTurn::fail("analyze boom"),
            ScriptedTurn::text("unused"),
        ]);
        let builder = BadCasePromptBuilder::new(model);

        let items: Vec<Result<String>> = builder
            .stream_build(
                "original prompt".to_owned(),
                vec![evaluated("c1", "too vague")],
                Language::ZhCn,
            )
            .collect()
            .await;

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].as_ref().unwrap_err().kind(), ErrorKind::Llm);
        assert_eq!(recorder.call_count(), 1);
    }

    #[tokio::test]
    async fn stream_build_yields_a_model_error_and_stops() {
        let (recorder, model) = scripted(vec![
            ScriptedTurn::text(ANALYZE_ANSWER),
            ScriptedTurn::fail("optimize boom"),
        ]);
        let builder = BadCasePromptBuilder::new(model);

        let items: Vec<Result<String>> = builder
            .stream_build(
                "original prompt".to_owned(),
                vec![evaluated("c1", "too vague")],
                Language::ZhCn,
            )
            .collect()
            .await;

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].as_ref().unwrap_err().kind(), ErrorKind::Llm);
        assert_eq!(recorder.call_count(), 2);
    }
}
