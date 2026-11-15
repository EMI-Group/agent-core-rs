//! Bounded `ReAct` (tool-execution) orchestration for the functional stage.
//!
//! Python's `skill_evaluator` built a `ReActAgent` subagent — model turn →
//! execute the requested sys-op tools → append the tool results → repeat until a
//! turn issues no tool call — and capped the loop with an iterations budget
//! (`MAX_ITERATIONS`, 25 in the SDK and 40 in `run_eval_query.py`). The Rust port
//! replaced that subagent with the [`CaseRunner`] seam, which left the functional
//! stage unable to execute tools at all; [`ReactCaseRunner`] is the bounded loop
//! on top of the existing typed seams, so functional scoring can run tools
//! without any transport.
//!
//! # The loop
//!
//! One [`EvalCase`] becomes a conversation: the system message is the configured
//! [`ReactCaseRunner::with_system_prompt`] or, by default, the skill document
//! (`SkillDefinition::to_markdown`); the user message is the case prompt, plus an
//! `Expected output: …` hint when the case carries a reference answer. Every
//! [`Model::invoke`] call offers the configured tool declarations (one per
//! registered [`ToolStubs`] entry when none were supplied). A turn without tool
//! calls ends the run and its text is the answer; a turn with tool calls has its
//! calls dispatched against the stubs in model order, recorded in a [`ToolTrace`],
//! and answered with one [`Role::Tool`](openjiuwen_foundation::prompt::Role::Tool)
//! message per call before the next model turn.
//!
//! # The cap
//!
//! [`max_rounds`](ReactCaseRunner::with_max_rounds) bounds how many *tool* rounds
//! run, so a model that never stops asking for tools costs at most
//! `max_rounds + 1` invocations. Once the budget is spent, the current turn's
//! text is accepted and [`ReactOutcome::truncated`] is set, so a caller can tell
//! a real answer from one the budget cut short (the tuning loop in
//! [`crate::tune`] cannot distinguish the two). A budget of `0` is rejected
//! before any model call.
//!
//! # Tool failures are data, not control flow
//!
//! An unknown tool, a stub handler that fails, and arguments that are not valid
//! JSON are all recorded as a failed [`ToolCallRecord`] whose error text is fed
//! back to the model as the tool result. The conversation continues and the run
//! still finishes, exactly like the tuning runner; only a model/transport failure
//! propagates ([`Error::llm`]).
//!
//! # Offline
//!
//! Everything runs in-process: [`ToolStubs`] are synchronous fixtures and a
//! [`Model`] can be an `openjiuwen_test_support::ScriptedModel`, so a tool-using
//! evaluation is deterministic with no network, clock, or environment read.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use openjiuwen_foundation::tool::ToolInfo;
use serde_json::{json, Value};

use crate::skill_creator::SkillDefinition;
use crate::tune::{ToolCallRecord, ToolStubs, ToolTrace};

use super::case::EvalCase;
use super::runner::CaseRunner;

/// How many tool rounds [`ReactCaseRunner`] runs before it accepts the model's
/// current text as the answer.
///
/// One round is one assistant turn that requests tools plus the tool results
/// answering it. The default matches the crate's tuning loop
/// ([`DEFAULT_MAX_TOOL_ROUNDS`](crate::tune::DEFAULT_MAX_TOOL_ROUNDS)) and sits
/// well below Python's 25/40 because offline stubs answer instantly, so a budget
/// of eight tool rounds is already generous for a fixture case.
pub const DEFAULT_MAX_REACT_ROUNDS: usize = 8;

/// Renders a tool result as the text of a tool-result message.
///
/// Same contract as the crate-private `render_tool_result` in
/// `crate::tune::evaluator::runner` (which is `pub(super)` there and therefore
/// unreachable from this module): a string result is rendered bare, so the model
/// reads the value instead of a quoted JSON literal; every other JSON value is
/// rendered compactly; and a value that cannot be serialized falls back to
/// `"null"` rather than failing the run.
fn render_tool_result(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "null".to_owned()),
    }
}

/// What one bounded `ReAct` run produced.
///
/// Unlike a bare answer, the outcome keeps the tool calls behind it, how many
/// tool rounds ran, and whether the budget stopped the loop early.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReactOutcome {
    /// The text of the turn that ended the run.
    pub answer: String,
    /// The tool calls executed while producing `answer`, in call order.
    pub trace: ToolTrace,
    /// How many completed tool rounds ran (`0` when the model never asked for a
    /// tool).
    pub rounds: usize,
    /// Whether [`max_rounds`](ReactCaseRunner::with_max_rounds) stopped the loop,
    /// so `answer` is the current turn's text rather than a model decision to
    /// stop.
    pub truncated: bool,
}

impl ReactOutcome {
    /// Whether the run executed at least one tool call, successful or failed.
    #[must_use]
    pub fn used_tools(&self) -> bool {
        !self.trace.is_empty()
    }

    /// Whether the run called `name`, successful or failed.
    #[must_use]
    pub fn used(&self, name: &str) -> bool {
        self.trace.used(name)
    }
}

/// A [`CaseRunner`] that runs each case through a bounded `ReAct` tool loop.
///
/// The model is consulted turn by turn: text ends the run, tool calls are
/// dispatched against the configured [`ToolStubs`] and fed back, and the round
/// budget bounds a model that never stops calling tools. See the module docs for
/// the conversation shape and the failure contract.
///
/// The runner is unusable only through misconfiguration: a `max_rounds` of `0`
/// is [`Error::validation`], as is a skill document that does not validate.
pub struct ReactCaseRunner {
    /// The model consulted for every turn of every case.
    model: Arc<dyn Model>,
    /// The tool fixtures the model's calls are dispatched to.
    stubs: ToolStubs,
    /// Explicit tool declarations; empty means "declare one per registered stub".
    declarations: Vec<Value>,
    /// How many tool rounds to run before accepting the model's text.
    max_rounds: usize,
    /// The system message; `None` means "use the skill document".
    system_prompt: Option<String>,
}

impl ReactCaseRunner {
    /// Creates a runner over `model`, without tool fixtures.
    ///
    /// The system message defaults to the skill document and the budget to
    /// [`DEFAULT_MAX_REACT_ROUNDS`].
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            stubs: ToolStubs::new(),
            declarations: Vec::new(),
            max_rounds: DEFAULT_MAX_REACT_ROUNDS,
            system_prompt: None,
        }
    }

    /// Sets the tool fixtures the model's calls are dispatched to.
    ///
    /// The registered names are also what the model is offered when no explicit
    /// declarations were supplied, so registering stubs is all a caller has to do.
    #[must_use]
    pub fn with_stubs(mut self, stubs: ToolStubs) -> Self {
        self.stubs = stubs;
        self
    }

    /// Overrides the tool declarations offered to the model.
    ///
    /// The values are the wire `{"type":"function","function":{…}}` shape — see
    /// [`ToolInfo`] serialization — and are passed to
    /// [`ModelRequestConfig::with_tools`] unchanged. An empty list restores the
    /// default of one declaration per registered stub.
    #[must_use]
    pub fn with_tool_declarations(mut self, declarations: Vec<Value>) -> Self {
        self.declarations = declarations;
        self
    }

    /// Sets how many tool rounds to run before accepting the model's text.
    ///
    /// A budget of `0` is rejected by [`run_traced`](Self::run_traced) and
    /// [`run`](CaseRunner::run) before any model call.
    #[must_use]
    pub fn with_max_rounds(mut self, max_rounds: usize) -> Self {
        self.max_rounds = max_rounds;
        self
    }

    /// Sets the system message, replacing the default skill document.
    #[must_use]
    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// The model this runner consults.
    #[must_use]
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    /// The tool fixtures this runner dispatches to.
    #[must_use]
    pub fn stubs(&self) -> &ToolStubs {
        &self.stubs
    }

    /// How many tool rounds this runner may run.
    #[must_use]
    pub fn max_rounds(&self) -> usize {
        self.max_rounds
    }

    /// Builds the request, declaring the tools the model may call.
    ///
    /// Explicit declarations win; otherwise one function declaration per
    /// registered stub name is generated with an empty object schema, so a caller
    /// never has to spell the wire shape out.
    fn request(&self) -> Result<ModelRequestConfig> {
        let declarations = if self.declarations.is_empty() {
            self.stub_declarations()?
        } else {
            self.declarations.clone()
        };
        Ok(ModelRequestConfig::new(self.model.model_name()).with_tools(declarations))
    }

    /// One `{"type":"function","function":{…}}` declaration per registered stub.
    fn stub_declarations(&self) -> Result<Vec<Value>> {
        let parameters = json!({"type": "object", "properties": {}});
        let mut declarations = Vec::with_capacity(self.stubs.len());
        for name in self.stubs.names() {
            let info = ToolInfo::new(
                name,
                format!("The `{name}` tool registered for this evaluation case."),
                parameters.clone(),
            );
            let declaration = serde_json::to_value(&info).map_err(|error| {
                Error::validation(format!(
                    "tool `{name}` could not be declared to the model: {error}"
                ))
            })?;
            declarations.push(declaration);
        }
        Ok(declarations)
    }

    /// The system message: the configured prompt or the skill document.
    fn system_message(&self, skill: &SkillDefinition) -> Result<String> {
        match &self.system_prompt {
            Some(prompt) => Ok(prompt.clone()),
            None => skill.to_markdown(),
        }
    }

    /// The user message: the case prompt plus its reference answer, when any.
    ///
    /// The expected output is rendered as an `Expected output: …` hint instead of
    /// re-encoding the whole case, so the message stays readable and stable.
    fn user_message(case: &EvalCase) -> String {
        let expected = case.expected_output.trim();
        if expected.is_empty() {
            case.prompt.clone()
        } else {
            format!("{}\n\nExpected output: {expected}", case.prompt)
        }
    }

    /// Runs one case through the bounded loop and reports the answer with its
    /// trace, round count, and truncation flag.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when [`max_rounds`](Self::with_max_rounds) is `0`
    /// (checked before any model call) or when the skill document does not
    /// validate; [`Error::llm`] when the model call fails or returns no choices.
    pub async fn run_traced(
        &self,
        case: &EvalCase,
        skill: &SkillDefinition,
    ) -> Result<ReactOutcome> {
        if self.max_rounds == 0 {
            return Err(Error::validation(
                "ReactCaseRunner.max_rounds must be at least 1",
            ));
        }
        let request = self.request()?;
        let mut messages = vec![
            BaseMessage::system(self.system_message(skill)?),
            BaseMessage::user(Self::user_message(case)),
        ];
        let mut trace = ToolTrace::default();
        let mut rounds = 0_usize;
        loop {
            let completion = self.model.invoke(&messages, &request).await?;
            let Some(message) = completion.message() else {
                return Err(Error::llm(format!(
                    "react runner case {:?}: model returned no choices; expected an answer message",
                    case.id
                )));
            };
            if message.tool_calls.is_empty() {
                return Ok(ReactOutcome {
                    answer: message.text().into_owned(),
                    trace,
                    rounds,
                    truncated: false,
                });
            }
            if rounds >= self.max_rounds {
                return Ok(ReactOutcome {
                    answer: message.text().into_owned(),
                    trace,
                    rounds,
                    truncated: true,
                });
            }
            rounds += 1;
            messages.push(message.clone());
            for call in &message.tool_calls {
                let rendered = match call.arguments_value() {
                    Ok(arguments) => match self.stubs.call(&call.name, &arguments) {
                        Ok(result) => {
                            trace.push(ToolCallRecord::success(
                                call.name.clone(),
                                arguments,
                                result.clone(),
                            ));
                            render_tool_result(&result)
                        }
                        Err(error) => {
                            let text = error.to_string();
                            trace.push(ToolCallRecord::failure(
                                call.name.clone(),
                                arguments,
                                text.clone(),
                            ));
                            text
                        }
                    },
                    Err(error) => {
                        let text = format!("invalid tool arguments: {error}");
                        trace.push(ToolCallRecord::failure(
                            call.name.clone(),
                            Value::Null,
                            text.clone(),
                        ));
                        text
                    }
                };
                let call_id = call
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("call_{}", trace.len()));
                messages.push(BaseMessage::tool(call_id, rendered));
            }
        }
    }
}

#[allow(clippy::missing_fields_in_debug)] // `declarations` and `system_prompt` are configuration, not identity
impl fmt::Debug for ReactCaseRunner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReactCaseRunner")
            .field("provider", &self.model.provider())
            .field("model_name", &self.model.model_name())
            .field("stubs", &self.stubs.names())
            .field("max_rounds", &self.max_rounds)
            .finish()
    }
}

#[async_trait]
impl CaseRunner for ReactCaseRunner {
    async fn run(&self, case: &EvalCase, skill: &SkillDefinition) -> Result<String> {
        Ok(self.run_traced(case, skill).await?.answer)
    }
}

#[cfg(test)]
mod tests {
    use openjiuwen_foundation::prompt::Role;
    use openjiuwen_test_support::{MockCompletion, MockToolCall, ScriptedModel};
    use serde_json::json;

    use super::*;
    use crate::skill_evaluator::case::CaseStatus;
    use crate::skill_evaluator::evaluator::SkillEvaluator;
    use crate::tune::ToolStub;

    /// The skill under test.
    fn skill() -> SkillDefinition {
        SkillDefinition::new(
            "demo",
            "Looks things up and answers",
            "# Demo\n\n## Output\n\nAn answer.",
        )
    }

    /// One case with the given expectation.
    fn case(expectation: &str) -> EvalCase {
        EvalCase::new("c1", "Case c1", "What is the answer?").with_expectation(expectation)
    }

    /// A table with the `lookup` stub answering `{"city": "Paris"}`.
    fn lookup_stubs() -> ToolStubs {
        ToolStubs::new().with_stub(
            ToolStub::constant("lookup", json!({"city": "Paris"})).expect("a non-blank stub name"),
        )
    }

    #[tokio::test]
    async fn runs_one_tool_round_then_answers() {
        let model = ScriptedModel::builder()
            .completion(MockCompletion::tool_calls(vec![MockToolCall::new(
                "lookup",
                json!({"q": "x"}),
            )
            .with_id("call_1")]))
            .respond_text("The capital is Paris.")
            .build();
        let recorder = model.clone();
        let runner = ReactCaseRunner::new(Arc::new(model)).with_stubs(lookup_stubs());

        let outcome = runner
            .run_traced(&case("contains:Paris"), &skill())
            .await
            .expect("the loop finishes with the model's answer");

        assert_eq!(outcome.answer, "The capital is Paris.");
        assert_eq!(outcome.rounds, 1);
        assert!(!outcome.truncated);
        assert!(outcome.used_tools());
        assert!(outcome.used("lookup"));
        assert_eq!(outcome.trace.count("lookup"), 1);
        let record = &outcome.trace.calls()[0];
        assert_eq!(record.arguments, json!({"q": "x"}));
        assert_eq!(record.result, json!({"city": "Paris"}));
        assert!(record.error.is_none(), "{record:?}");

        let calls = recorder.calls();
        assert_eq!(calls.len(), 2);
        let expected = json!([{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "The `lookup` tool registered for this evaluation case.",
                "parameters": {"type": "object", "properties": {}}
            }
        }]);
        assert_eq!(
            calls[0].request.tools,
            Some(expected),
            "registered stubs are declared to the model"
        );
        let tool_message = calls[1]
            .messages
            .iter()
            .find(|message| message.role.is_tool())
            .expect("the tool result is fed back");
        assert_eq!(tool_message.role, Role::Tool);
        assert_eq!(tool_message.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(tool_message.text(), "{\"city\":\"Paris\"}");
    }

    #[tokio::test]
    async fn a_plain_turn_ends_the_run_without_tools() {
        let model = ScriptedModel::text("plain answer");
        let recorder = model.clone();
        let runner = ReactCaseRunner::new(Arc::new(model)).with_stubs(lookup_stubs());

        let outcome = runner
            .run_traced(&case("contains:plain"), &skill())
            .await
            .expect("a plain turn is the answer");

        assert_eq!(outcome.answer, "plain answer");
        assert_eq!(outcome.rounds, 0);
        assert!(!outcome.truncated);
        assert!(!outcome.used_tools());
        assert!(outcome.trace.is_empty());
        assert_eq!(recorder.call_count(), 1);
    }

    #[tokio::test]
    async fn executes_every_call_of_one_turn_in_model_order() {
        let model = ScriptedModel::builder()
            .completion(MockCompletion::tool_calls(vec![
                MockToolCall::new("lookup", json!({"n": 1})).with_id("call_a"),
                MockToolCall::new("lookup", json!({"n": 2})).with_id("call_b"),
            ]))
            .respond_text("done")
            .build();
        let recorder = model.clone();
        let runner = ReactCaseRunner::new(Arc::new(model)).with_stubs(lookup_stubs());

        let outcome = runner
            .run_traced(&case("contains:done"), &skill())
            .await
            .expect("both calls run in one round");

        assert_eq!(outcome.rounds, 1);
        assert_eq!(outcome.trace.len(), 2);
        assert_eq!(
            outcome.trace.names(),
            vec!["lookup", "lookup"],
            "records keep model order"
        );
        assert_eq!(outcome.trace.calls()[0].arguments, json!({"n": 1}));
        assert_eq!(outcome.trace.calls()[1].arguments, json!({"n": 2}));

        let calls = recorder.calls();
        let ids: Vec<Option<&str>> = calls[1]
            .messages
            .iter()
            .filter(|message| message.role.is_tool())
            .map(|message| message.tool_call_id.as_deref())
            .collect();
        assert_eq!(ids, vec![Some("call_a"), Some("call_b")]);
    }

    #[tokio::test]
    async fn records_an_unknown_tool_and_a_failing_stub_then_finishes() {
        let model = ScriptedModel::builder()
            .completion(MockCompletion::tool_calls(vec![
                MockToolCall::new("missing", json!({})).with_id("call_x"),
                MockToolCall::new("boom", json!({"q": "x"})).with_id("call_y"),
            ]))
            .respond_text("recovered")
            .build();
        let recorder = model.clone();
        let stubs = ToolStubs::new().with_stub(
            ToolStub::new("boom", |_| Err(Error::validation("boom failed")))
                .expect("a non-blank stub name"),
        );
        let runner = ReactCaseRunner::new(Arc::new(model)).with_stubs(stubs);

        let outcome = runner
            .run_traced(&case("contains:recovered"), &skill())
            .await
            .expect("tool failures are recorded, not propagated");

        assert_eq!(outcome.answer, "recovered");
        assert_eq!(outcome.rounds, 1);
        assert!(!outcome.truncated);
        assert_eq!(outcome.trace.len(), 2);
        let unknown = &outcome.trace.calls()[0];
        assert!(unknown.is_error(), "{unknown:?}");
        let unknown_error = unknown.error.as_deref().unwrap_or_default();
        assert!(unknown_error.contains("missing"), "{unknown_error}");
        let failing = &outcome.trace.calls()[1];
        assert!(failing.is_error(), "{failing:?}");
        assert_eq!(failing.error.as_deref(), Some("validation: boom failed"));

        let calls = recorder.calls();
        let texts: Vec<String> = calls[1]
            .messages
            .iter()
            .filter(|message| message.role.is_tool())
            .map(|message| message.text().into_owned())
            .collect();
        assert_eq!(
            texts,
            vec![
                unknown_error.to_owned(),
                "validation: boom failed".to_owned()
            ],
            "both failures are fed back as tool results"
        );
    }

    #[tokio::test]
    async fn records_unparseable_arguments_as_a_failure() {
        let model = ScriptedModel::builder()
            .completion(MockCompletion::tool_calls(vec![MockToolCall::new(
                "lookup",
                json!({}),
            )
            .with_raw_arguments("{not json")]))
            .respond_text("continued")
            .build();
        let recorder = model.clone();
        let runner = ReactCaseRunner::new(Arc::new(model)).with_stubs(lookup_stubs());

        let outcome = runner
            .run_traced(&case("contains:continued"), &skill())
            .await
            .expect("a malformed call is recorded instead of panicking");

        assert_eq!(outcome.answer, "continued");
        assert_eq!(outcome.rounds, 1);
        let record = &outcome.trace.calls()[0];
        assert!(record.is_error(), "{record:?}");
        assert_eq!(record.arguments, Value::Null);
        let error = record.error.as_deref().unwrap_or_default();
        assert!(error.starts_with("invalid tool arguments:"), "{error}");

        let calls = recorder.calls();
        let message = calls[1]
            .messages
            .iter()
            .find(|message| message.role.is_tool())
            .expect("the failure is fed back");
        assert_eq!(message.text(), error);
    }

    #[tokio::test]
    async fn the_round_budget_stops_a_tool_looping_model() {
        /// The turn a looping model keeps repeating.
        fn looping() -> MockCompletion {
            MockCompletion::tool_call("lookup", json!({"q": "x"}))
        }

        let model = ScriptedModel::builder()
            .completion(looping())
            .completion(looping())
            .completion(looping())
            .build();
        let recorder = model.clone();
        let runner = ReactCaseRunner::new(Arc::new(model))
            .with_stubs(lookup_stubs())
            .with_max_rounds(2);

        let outcome = runner
            .run_traced(&case("contains:x"), &skill())
            .await
            .expect("the budget ends the loop");

        assert_eq!(
            recorder.call_count(),
            3,
            "two tool rounds plus the truncated turn"
        );
        assert_eq!(outcome.rounds, 2);
        assert_eq!(outcome.trace.len(), 2);
        assert!(outcome.truncated, "the budget stopped the loop");
        assert_eq!(
            outcome.answer, "",
            "the exhausted turn's empty text is accepted"
        );
    }

    #[tokio::test]
    async fn a_zero_budget_is_rejected_before_any_model_call() {
        let model = ScriptedModel::text("unused");
        let recorder = model.clone();
        let runner = ReactCaseRunner::new(Arc::new(model)).with_max_rounds(0);

        let error = runner
            .run_traced(&case("contains:x"), &skill())
            .await
            .expect_err("a zero budget is rejected");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("max_rounds"), "{error}");
        assert_eq!(recorder.call_count(), 0);

        let runner_error = runner
            .run(&case("contains:x"), &skill())
            .await
            .expect_err("run inherits the same check");
        assert_eq!(
            runner_error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert_eq!(recorder.call_count(), 0);
    }

    #[tokio::test]
    async fn a_custom_system_prompt_and_expected_output_shape_the_conversation() {
        let model = ScriptedModel::text("answer");
        let recorder = model.clone();
        let runner = ReactCaseRunner::new(Arc::new(model)).with_system_prompt("be terse");
        let case = case("contains:answer").with_expected_output("  answer  ");

        runner
            .run_traced(&case, &skill())
            .await
            .expect("the run finishes");

        let calls = recorder.calls();
        let messages = &calls[0].messages;
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].text(), "be terse");
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(
            messages[1].text(),
            "What is the answer?\n\nExpected output: answer"
        );
    }

    #[tokio::test]
    async fn the_skill_document_is_the_default_system_message() {
        let model = ScriptedModel::text("answer");
        let recorder = model.clone();
        let runner = ReactCaseRunner::new(Arc::new(model));

        runner
            .run_traced(&case("contains:answer"), &skill())
            .await
            .expect("the run finishes");

        let calls = recorder.calls();
        let system = calls[0].messages[0].text().into_owned();
        assert!(system.contains("demo"), "{system}");
        assert!(system.contains("An answer."), "{system}");
        assert_eq!(
            calls[0].request.tools, None,
            "no stubs means no declarations"
        );
    }

    #[tokio::test]
    async fn explicit_declarations_override_the_stub_derived_ones() {
        let model = ScriptedModel::text("answer");
        let recorder = model.clone();
        let custom = json!({
            "type": "function",
            "function": {"name": "custom", "description": "d", "parameters": {"type": "object"}}
        });
        let runner = ReactCaseRunner::new(Arc::new(model))
            .with_stubs(lookup_stubs())
            .with_tool_declarations(vec![custom.clone()]);

        runner
            .run_traced(&case("contains:answer"), &skill())
            .await
            .expect("the run finishes");

        assert_eq!(recorder.calls()[0].request.tools, Some(json!([custom])));
    }

    #[tokio::test]
    async fn the_runner_is_a_case_runner() {
        let model = ScriptedModel::builder()
            .completion(MockCompletion::tool_call("lookup", json!({"q": "paris"})))
            .respond_text("The capital of France is Paris.")
            .build();
        let runner = ReactCaseRunner::new(Arc::new(model)).with_stubs(lookup_stubs());
        assert_eq!(
            CaseRunner::run(&runner, &case("contains:Paris"), &skill())
                .await
                .expect("the runner answers"),
            "The capital of France is Paris."
        );
    }

    #[tokio::test]
    async fn the_evaluator_runs_a_tool_using_case_offline() {
        let model = ScriptedModel::builder()
            .completion(MockCompletion::tool_call("lookup", json!({"q": "paris"})))
            .respond_text("The capital of France is Paris.")
            .respond_text("no idea")
            .build();
        let runner = ReactCaseRunner::new(Arc::new(model)).with_stubs(lookup_stubs());
        let evaluator = SkillEvaluator::new(Arc::new(runner));
        let cases = [
            case("contains:Paris"),
            EvalCase::new("c2", "Case c2", "What is the capital of Peru?")
                .with_expectation("contains:Lima"),
        ];

        let report = evaluator
            .evaluate_at(&skill(), &cases, 1_700_000_000_000)
            .await
            .expect("a tool-backed run evaluates");

        assert_eq!(report.functional.results[0].status, CaseStatus::Passed);
        assert_eq!(report.functional.results[1].status, CaseStatus::Failed);
        assert_eq!(report.functional.summary.total, 2);
        assert_eq!(report.functional.summary.passed, 1);
        assert!(
            (report.functional.summary.pass_rate - 0.5).abs() < 1e-9,
            "the assertion-based pass rate is unchanged: {}",
            report.functional.summary.pass_rate
        );
    }

    #[test]
    fn debug_names_the_model_stubs_and_budget() {
        let runner = ReactCaseRunner::new(Arc::new(ScriptedModel::text("x")))
            .with_stubs(lookup_stubs())
            .with_max_rounds(3);
        let rendered = format!("{runner:?}");
        assert!(rendered.contains("ReactCaseRunner"), "{rendered}");
        assert!(rendered.contains("scripted-model"), "{rendered}");
        assert!(rendered.contains("lookup"), "{rendered}");
        assert!(rendered.contains('3'), "{rendered}");
        assert_eq!(runner.max_rounds(), 3);
        assert_eq!(runner.stubs().names(), vec!["lookup"]);
        assert_eq!(runner.model().model_name(), "scripted-model");
    }

    #[test]
    fn outcome_defaults_and_accessors() {
        let outcome = ReactOutcome::default();
        assert!(!outcome.used_tools());
        assert!(!outcome.used("lookup"));
        assert_eq!(outcome.rounds, 0);
        assert!(!outcome.truncated);

        let outcome = ReactOutcome {
            answer: "a".to_owned(),
            trace: ToolTrace::new([ToolCallRecord::success("lookup", json!({}), json!(1))]),
            rounds: 1,
            truncated: true,
        };
        assert!(outcome.used_tools());
        assert!(outcome.used("lookup"));
        assert!(!outcome.used("other"));
    }

    #[test]
    fn the_default_budget_matches_the_tuning_loop() {
        assert_eq!(
            DEFAULT_MAX_REACT_ROUNDS,
            crate::tune::DEFAULT_MAX_TOOL_ROUNDS
        );
        assert_eq!(
            ReactCaseRunner::new(Arc::new(ScriptedModel::text("x"))).max_rounds(),
            DEFAULT_MAX_REACT_ROUNDS
        );
    }
}
