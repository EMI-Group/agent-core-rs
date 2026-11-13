//! The runner seam: turn `(candidate prompt, case)` into an answer map.
//!
//! A runner may also execute the tools a case declares. [`ModelPromptRunner`]
//! does so through [`ToolStubs`]: every tool call the model requests is executed
//! in order, recorded in a [`ToolTrace`] — failures included — and fed back to
//! the model as a tool-result message, so the conversation continues until the
//! model answers in text or the `max_tool_rounds` budget is exhausted.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tune::base::Case;
use crate::tune::tool::{RunOutcome, ToolCallRecord, ToolStubs, ToolTrace};
use crate::util::extract_json_fence;

/// How many tool rounds [`ModelPromptRunner`] runs before it accepts the
/// model's current text as the answer.
///
/// One round is one assistant turn that requests tools plus the tool results
/// answering it; the budget bounds a model that never stops calling tools.
pub const DEFAULT_MAX_TOOL_ROUNDS: usize = 8;

/// Renders one case's inputs as the user-facing text of a prompt.
///
/// One `[key]: value` line per input, in the input map's deterministic key
/// order. String values are rendered bare; every other JSON value is rendered
/// compactly, so a number stays `4` and an object stays `{"a":1}`.
///
/// # Examples
///
/// ```
/// # use std::collections::BTreeMap;
/// # use openjiuwen_dev_tools::tune::{render_case_inputs, Case};
/// # use serde_json::Value;
/// let mut inputs = BTreeMap::new();
/// inputs.insert("question".to_owned(), Value::String("2 + 2?".to_owned()));
/// let mut label = BTreeMap::new();
/// label.insert("answer".to_owned(), Value::String("4".to_owned()));
/// let case = Case::new(inputs, label)?;
/// assert_eq!(render_case_inputs(&case), "[question]: 2 + 2?");
/// # Ok::<(), openjiuwen_common::error::Error>(())
/// ```
#[must_use]
pub fn render_case_inputs(case: &Case) -> String {
    case.inputs
        .iter()
        .map(|(key, value)| format!("[{key}]: {}", render_value(value)))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Renders a JSON value as plain text: strings unquoted, everything else as
/// compact JSON.
pub(super) fn render_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Renders a tool result as the text of a tool-result message.
///
/// A string result is rendered bare, so the model reads the value instead of a
/// quoted JSON literal; every other JSON value is rendered compactly. A value
/// that cannot be serialized falls back to `null` rather than failing the run.
pub(super) fn render_tool_result(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "null".to_owned()),
    }
}

/// Turns a raw model answer into an answer map.
///
/// A fenced JSON object or a whole-answer JSON object is used as the map; any
/// other answer becomes `{"output": <trimmed text>}`.
pub(super) fn answer_to_map(answer: &str) -> BTreeMap<String, Value> {
    if let Some(Value::Object(object)) = extract_json_fence(answer) {
        return object.into_iter().collect();
    }
    if let Ok(Value::Object(object)) = serde_json::from_str::<Value>(answer.trim()) {
        return object.into_iter().collect();
    }
    let mut map = BTreeMap::new();
    map.insert("output".to_owned(), Value::String(answer.trim().to_owned()));
    map
}

/// Runs a candidate prompt over one case and returns the answer map.
#[async_trait]
pub trait PromptRunner: Send + Sync {
    /// Runs `prompt` for `case`.
    ///
    /// # Errors
    ///
    /// Implementation-defined: a model runner propagates the provider error,
    /// a scripted runner reports an unknown case.
    async fn run(&self, prompt: &str, case: &Case) -> Result<BTreeMap<String, Value>>;

    /// Runs `prompt` for `case` and reports the tool calls behind the answer.
    ///
    /// The default runs [`run`](Self::run) and reports an empty trace, so a
    /// runner that executes no tools needs no change.
    ///
    /// # Errors
    ///
    /// Whatever [`run`](Self::run) returns.
    async fn run_traced(&self, prompt: &str, case: &Case) -> Result<RunOutcome> {
        Ok(RunOutcome::new(self.run(prompt, case).await?))
    }
}

/// A [`PromptRunner`] that calls a [`Model`] and executes the tools it asks for.
///
/// The candidate prompt is the system message and the rendered case inputs
/// ([`render_case_inputs`]) the user message; the tools the case declares are
/// offered to the model in the request. The answer becomes an answer map: a
/// fenced or whole-answer JSON object is used as-is, any other answer becomes
/// `{"output": <trimmed text>}`.
///
/// When the model requests tools, each call runs against the configured
/// [`ToolStubs`] in the model's order and its outcome is recorded in a
/// [`ToolTrace`]. A failing stub — and equally a call to a tool with no stub
/// registered at all, which reports [`Error::not_found`] — is recorded with its
/// error and the error text is fed back to the model, so the conversation keeps
/// going instead of the failure being swallowed or propagated. The loop is
/// bounded by [`max_tool_rounds`](Self::with_max_tool_rounds): once that many
/// rounds ran, the model's current text is accepted as the answer (usually the
/// empty text of a tool-calling turn).
pub struct ModelPromptRunner {
    /// The model consulted for every case.
    model: Arc<dyn Model>,
    /// The tool fixtures the model's calls are dispatched to.
    stubs: ToolStubs,
    /// How many tool rounds to run before accepting the model's text.
    max_tool_rounds: usize,
}

impl ModelPromptRunner {
    /// Creates a runner over `model`, without tool fixtures.
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            stubs: ToolStubs::new(),
            max_tool_rounds: DEFAULT_MAX_TOOL_ROUNDS,
        }
    }

    /// Sets the tool fixtures the model's calls are dispatched to.
    #[must_use]
    pub fn with_stubs(mut self, stubs: ToolStubs) -> Self {
        self.stubs = stubs;
        self
    }

    /// Sets how many tool rounds to run before accepting the model's text.
    ///
    /// A budget of `0` is rejected by [`run`](PromptRunner::run) and
    /// [`run_traced`](PromptRunner::run_traced) before any model call.
    #[must_use]
    pub fn with_max_tool_rounds(mut self, max_tool_rounds: usize) -> Self {
        self.max_tool_rounds = max_tool_rounds;
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
    pub fn max_tool_rounds(&self) -> usize {
        self.max_tool_rounds
    }

    /// Builds the request, declaring the case's tools when it has any.
    fn request_for(&self, case: &Case) -> Result<ModelRequestConfig> {
        let request = ModelRequestConfig::new(self.model.model_name());
        let Some(tools) = case.tools().filter(|tools| !tools.is_empty()) else {
            return Ok(request);
        };
        let mut declarations = Vec::with_capacity(tools.len());
        for tool in tools {
            let declaration = serde_json::to_value(tool).map_err(|error| {
                Error::validation(format!(
                    "tool `{}` could not be declared to the model: {error}",
                    tool.function.name
                ))
            })?;
            declarations.push(declaration);
        }
        Ok(request.with_tools(declarations))
    }
}

impl fmt::Debug for ModelPromptRunner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelPromptRunner")
            .field("provider", &self.model.provider())
            .field("model_name", &self.model.model_name())
            .field("stubs", &self.stubs.names())
            .field("max_tool_rounds", &self.max_tool_rounds)
            .finish()
    }
}

#[async_trait]
impl PromptRunner for ModelPromptRunner {
    async fn run(&self, prompt: &str, case: &Case) -> Result<BTreeMap<String, Value>> {
        Ok(self.run_traced(prompt, case).await?.answer)
    }

    async fn run_traced(&self, prompt: &str, case: &Case) -> Result<RunOutcome> {
        if self.max_tool_rounds == 0 {
            return Err(Error::validation(
                "ModelPromptRunner.max_tool_rounds must be at least 1",
            ));
        }
        let request = self.request_for(case)?;
        let mut messages = vec![
            BaseMessage::system(prompt.to_owned()),
            BaseMessage::user(render_case_inputs(case)),
        ];
        let mut trace = ToolTrace::default();
        let mut rounds = 0_usize;
        loop {
            let completion = self.model.invoke(&messages, &request).await?;
            let Some(message) = completion.message() else {
                return Err(Error::llm(
                    "prompt runner model returned no choices; expected an answer message",
                ));
            };
            if message.tool_calls.is_empty() {
                return Ok(RunOutcome::with_trace(
                    answer_to_map(&message.text()),
                    trace,
                ));
            }
            if rounds >= self.max_tool_rounds {
                return Ok(RunOutcome::with_trace(
                    answer_to_map(&message.text()),
                    trace,
                ));
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

/// Locks `mutex`, recovering the guard when a previous holder panicked.
///
/// A poisoned lock only means some other caller panicked; the recorded calls are
/// still consistent, so recovering keeps this helper panic-free.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// One recorded [`ScriptedPromptRunner`] call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PromptCall {
    /// The case the call evaluated.
    pub case_id: String,
    /// The candidate prompt that was evaluated.
    pub prompt: String,
}

/// A [`PromptRunner`] that replays canned answers keyed by case id.
///
/// Unknown case ids fall back to [`with_fallback`](Self::with_fallback) when it
/// is set and report [`Error::not_found`] otherwise. Every call is recorded, so
/// a test can assert which prompts reached the runner and in which order.
#[derive(Debug, Default)]
pub struct ScriptedPromptRunner {
    /// Answers keyed by case id.
    answers: BTreeMap<String, BTreeMap<String, Value>>,
    /// Answer used for an unscripted case, when any.
    fallback: Option<BTreeMap<String, Value>>,
    /// Calls in call order.
    calls: Mutex<Vec<PromptCall>>,
}

impl ScriptedPromptRunner {
    /// Creates a runner without answers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Answers `case_id` with `answer`.
    #[must_use]
    pub fn with_answer(
        mut self,
        case_id: impl Into<String>,
        answer: BTreeMap<String, Value>,
    ) -> Self {
        self.answers.insert(case_id.into(), answer);
        self
    }

    /// Answers `case_id` with a single `output` entry holding `text`.
    #[must_use]
    pub fn with_text_answer(self, case_id: impl Into<String>, text: impl Into<String>) -> Self {
        let mut answer = BTreeMap::new();
        answer.insert("output".to_owned(), Value::String(text.into()));
        self.with_answer(case_id, answer)
    }

    /// Answers every unscripted case with `answer`.
    #[must_use]
    pub fn with_fallback(mut self, answer: BTreeMap<String, Value>) -> Self {
        self.fallback = Some(answer);
        self
    }

    /// The calls recorded so far, oldest first.
    ///
    /// # Panics
    ///
    /// Panics only when the internal lock was poisoned by a panicking caller.
    #[must_use]
    pub fn calls(&self) -> Vec<PromptCall> {
        lock(&self.calls).clone()
    }

    /// How many calls were recorded.
    #[must_use]
    pub fn call_count(&self) -> usize {
        lock(&self.calls).len()
    }
}

#[async_trait]
impl PromptRunner for ScriptedPromptRunner {
    async fn run(&self, prompt: &str, case: &Case) -> Result<BTreeMap<String, Value>> {
        lock(&self.calls).push(PromptCall {
            case_id: case.case_id.clone(),
            prompt: prompt.to_owned(),
        });
        if let Some(answer) = self.answers.get(&case.case_id) {
            return Ok(answer.clone());
        }
        self.fallback.clone().ok_or_else(|| {
            Error::not_found(format!(
                "scripted prompt runner has no answer for case {:?}",
                case.case_id
            ))
        })
    }
}
