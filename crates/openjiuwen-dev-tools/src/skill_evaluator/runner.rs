//! Case execution: the runner seam and its scripted and model-backed runs.
//!
//! A [`CaseRunner`] turns one [`EvalCase`] plus the skill under test into the
//! answer the assertions are checked against. Two implementations cover the
//! offline and model-driven cases:
//!
//! - [`ScriptedCaseRunner`] serves canned answers keyed by case id, so tests and
//!   deterministic pipelines never touch a model.
//! - [`ModelCaseRunner`] sends one system message (the skill document) plus the
//!   case prompt to a [`Model`] and returns the
//!   answer text.
//!
//! Python ran cases through a shell harness; the Rust port replaces that with
//! this trait so an evaluator can run the same assertions over scripted answers
//! or a real model.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;

use crate::skill_creator::SkillDefinition;

use super::case::EvalCase;

/// Produces the answer of one evaluation case.
///
/// The runner receives the skill under test so a model-backed implementation can
/// frame the conversation with the skill document; implementations that serve
/// canned answers may ignore it.
#[async_trait]
pub trait CaseRunner: Send + Sync {
    /// Runs one case against `skill` and returns the answer text.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// when the runner has no answer for the case id, and
    /// [`Error::llm`](openjiuwen_common::error::Error::llm) when a model call
    /// fails or returns no usable text.
    async fn run(&self, case: &EvalCase, skill: &SkillDefinition) -> Result<String>;
}

/// A [`CaseRunner`] that serves canned answers keyed by case id.
///
/// The runner is a plain map, so it is cheap to clone and compare in tests. A
/// case whose id was not registered is a validation error, never an empty answer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScriptedCaseRunner {
    /// Canned answers keyed by case id.
    responses: BTreeMap<String, String>,
}

impl ScriptedCaseRunner {
    /// Creates an empty runner.
    #[must_use]
    pub fn new() -> Self {
        Self {
            responses: BTreeMap::new(),
        }
    }

    /// Registers the answer for `case_id`, replacing any previous one.
    #[must_use]
    pub fn respond(mut self, case_id: impl Into<String>, output: impl Into<String>) -> Self {
        self.responses.insert(case_id.into(), output.into());
        self
    }

    /// Registers each case's expected output as its scripted answer.
    ///
    /// This makes reference answers available without spelling every response
    /// out; cases with a blank expected output still get an empty answer.
    #[must_use]
    pub fn from_expected_outputs(cases: &[EvalCase]) -> Self {
        let mut responses = BTreeMap::new();
        for case in cases {
            responses.insert(case.id.clone(), case.expected_output.clone());
        }
        Self { responses }
    }

    /// The registered answers, keyed by case id.
    #[must_use]
    pub fn responses(&self) -> &BTreeMap<String, String> {
        &self.responses
    }
}

#[async_trait]
impl CaseRunner for ScriptedCaseRunner {
    async fn run(&self, case: &EvalCase, _skill: &SkillDefinition) -> Result<String> {
        self.responses.get(&case.id).cloned().ok_or_else(|| {
            Error::validation(format!(
                "no scripted response for evaluation case {:?}",
                case.id
            ))
        })
    }
}

/// A [`CaseRunner`] that asks a model to answer each case.
///
/// Every case is one non-streaming [`Model::invoke`] call: a system message
/// carrying the skill name, description, and instructions, followed by the case
/// prompt as the user message. The first choice's text is the answer.
pub struct ModelCaseRunner {
    /// The model consulted for every case.
    model: Arc<dyn Model>,
}

impl ModelCaseRunner {
    /// Creates a runner over `model`.
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self { model }
    }
}

#[async_trait]
impl CaseRunner for ModelCaseRunner {
    async fn run(&self, case: &EvalCase, skill: &SkillDefinition) -> Result<String> {
        let system = format!(
            "You are executing the skill {:?}. Answer the user's request by following the \
             skill document below.\n\n{}\n\n{}\n\n{}",
            skill.name, skill.name, skill.description, skill.instructions
        );
        let messages = [
            BaseMessage::system(system),
            BaseMessage::user(case.prompt.clone()),
        ];
        let request = ModelRequestConfig::new(self.model.model_name());
        let response = self.model.invoke(&messages, &request).await?;
        let answer = response
            .message()
            .map(|message| message.text().into_owned())
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| {
                Error::llm(format!(
                    "evaluation case {:?}: model returned no answer text",
                    case.id
                ))
            })?;
        Ok(answer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill() -> SkillDefinition {
        SkillDefinition::new(
            "demo",
            "Summarize documents",
            "# Demo\n\n## Output\n\nA summary.",
        )
    }

    fn case(id: &str) -> EvalCase {
        EvalCase::new(id, format!("Case {id}"), format!("Prompt {id}"))
            .with_expectation("contains:answer")
    }

    #[tokio::test]
    async fn scripted_runner_serves_registered_answers() {
        let runner = ScriptedCaseRunner::new()
            .respond("case-1", "first answer")
            .respond("case-2", "second answer");
        assert_eq!(runner.responses().len(), 2);
        assert_eq!(
            runner.run(&case("case-1"), &skill()).await.expect("run"),
            "first answer"
        );
        assert_eq!(
            runner.run(&case("case-2"), &skill()).await.expect("run"),
            "second answer"
        );
    }

    #[tokio::test]
    async fn scripted_runner_rejects_unknown_case() {
        let runner = ScriptedCaseRunner::new();
        let error = runner
            .run(&case("missing"), &skill())
            .await
            .expect_err("missing response");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("missing"));
    }

    #[test]
    fn scripted_runner_builds_from_expected_outputs() {
        let cases = [
            case("case-1").with_expected_output("expected one"),
            case("case-2").with_expected_output("expected two"),
        ];
        let runner = ScriptedCaseRunner::from_expected_outputs(&cases);
        assert_eq!(runner.responses()["case-1"], "expected one");
        assert_eq!(runner.responses()["case-2"], "expected two");
        assert_eq!(runner.responses().len(), 2);
    }

    #[test]
    fn scripted_runner_default_is_empty() {
        assert_eq!(ScriptedCaseRunner::default(), ScriptedCaseRunner::new());
        assert_eq!(ScriptedCaseRunner::default().responses().len(), 0);
    }

    #[tokio::test]
    async fn model_runner_returns_the_answer_text() {
        let model = Arc::new(openjiuwen_test_support::ScriptedModel::text("answer text"));
        let runner = ModelCaseRunner::new(model.clone());
        let case = case("case-1");
        let skill = skill();

        let answer = runner.run(&case, &skill).await.expect("run");
        assert_eq!(answer, "answer text");
        assert_eq!(model.call_count(), 1);

        let calls = model.calls();
        let messages = &calls[0].messages;
        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[0].role,
            openjiuwen_foundation::prompt::Role::System
        );
        let system = messages[0].text();
        assert!(system.contains("demo"), "{system}");
        assert!(system.contains("Summarize documents"), "{system}");
        assert!(system.contains("A summary."), "{system}");
        assert_eq!(messages[1].role, openjiuwen_foundation::prompt::Role::User);
        assert_eq!(messages[1].text().as_ref(), case.prompt.as_str());
    }

    #[tokio::test]
    async fn model_runner_rejects_an_empty_answer() {
        let model = Arc::new(openjiuwen_test_support::ScriptedModel::text(""));
        let runner = ModelCaseRunner::new(model);
        let error = runner
            .run(&case("case-1"), &skill())
            .await
            .expect_err("empty answer");
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Llm);
        assert!(error.to_string().contains("case-1"));
    }

    #[tokio::test]
    async fn model_runner_rejects_a_whitespace_answer() {
        let model = Arc::new(openjiuwen_test_support::ScriptedModel::text("  \n\t "));
        let runner = ModelCaseRunner::new(model);
        let error = runner
            .run(&case("case-1"), &skill())
            .await
            .expect_err("whitespace answer");
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Llm);
    }
}
