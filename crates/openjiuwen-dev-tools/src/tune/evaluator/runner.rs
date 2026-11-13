//! The runner seam: turn `(candidate prompt, case)` into an answer map.

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
use crate::util::extract_json_fence;

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
}

/// A [`PromptRunner`] that calls a [`Model`] once per case.
///
/// The candidate prompt is the system message and the rendered case inputs
/// ([`render_case_inputs`]) the user message. The answer becomes an answer map:
/// a fenced or whole-answer JSON object is used as-is, any other answer becomes
/// `{"output": <trimmed text>}`.
pub struct ModelPromptRunner {
    /// The model consulted for every case.
    model: Arc<dyn Model>,
}

impl ModelPromptRunner {
    /// Creates a runner over `model`.
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self { model }
    }

    /// The model this runner consults.
    #[must_use]
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }
}

impl fmt::Debug for ModelPromptRunner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelPromptRunner")
            .field("provider", &self.model.provider())
            .field("model_name", &self.model.model_name())
            .finish()
    }
}

#[async_trait]
impl PromptRunner for ModelPromptRunner {
    async fn run(&self, prompt: &str, case: &Case) -> Result<BTreeMap<String, Value>> {
        let messages = [
            BaseMessage::system(prompt.to_owned()),
            BaseMessage::user(render_case_inputs(case)),
        ];
        let request = ModelRequestConfig::new(self.model.model_name());
        let completion = self.model.invoke(&messages, &request).await?;
        if completion.choices.is_empty() {
            return Err(Error::llm(
                "prompt runner model returned no choices; expected an answer message",
            ));
        }
        let answer = completion
            .message()
            .map(|message| message.text().into_owned())
            .unwrap_or_default();
        Ok(answer_to_map(&answer))
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
