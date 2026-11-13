//! Legacy reasoner: LLM intent detection and the placeholder planner.
//!
//! Ports `openjiuwen/core/controller/legacy/reasoner/intent_detector.py` and
//! `planner.py`. Python's detector prompts the model with a category list and
//! parses a `{"result": n}` answer; [`IntentDetector`] keeps that contract but
//! returns the category name instead of turning it into tasks (task generation
//! belongs to the intent-detection controller, which speaks the new pipeline's
//! [`Intent`](crate::controller::schema::Intent) type). [`LegacyPlanner`] is
//! Python's stub planner: one pending, undefined task per event, without
//! decomposition or dependency wiring.
use super::schema::{LegacyEvent, LegacyTask, TaskInput};
use openjiuwen_common::error::Result;
use openjiuwen_foundation::{BaseMessage, Model, ModelRequestConfig};
use serde_json::Value;
use std::sync::Arc;
use tracing::warn;
/// Default classification prompt.
///
/// Python ships a Chinese prompt; this port keeps the same instructions in
/// deterministic English: pick the option whose meaning matches the input, or
/// the default class when nothing fits, and answer with JSON only.
pub const INTENT_DETECTION_PROMPT: &str = "\
You are an intent classification assistant. Decide which category the user's input belongs to.
Choose the default class when the input has no clear intent or you cannot tell.
The allowed categories are listed with 1-based indices.
Answer with JSON only, without any explanation, in this exact shape: {\"result\": <int>}";
/// JSON key the model must answer the category index under.
const RESULT_KEY: &str = "result";
/// LLM category classifier (Python's `IntentDetector`).
///
/// The detector never fails a caller: a model error or an unparsable answer
/// falls back to the configured default class.
pub struct IntentDetector {
    /// Model the classification request is sent to.
    model: Arc<dyn Model>,
    /// Category names, in the order the prompt lists them.
    categories: Vec<String>,
    /// Category returned when classification does not match.
    default_class: String,
    /// Request configuration of the model call.
    request: ModelRequestConfig,
    /// System prompt sent to the model.
    prompt: String,
}
impl IntentDetector {
    /// Category Python's default `IntentDetectionConfig` uses when nothing
    /// matches.
    pub const DEFAULT_CLASS: &'static str = "unknown";
    /// Builds a detector over `model` and the ordered `categories`.
    #[must_use]
    pub fn new(model: Arc<dyn Model>, categories: Vec<String>) -> Self {
        let request = ModelRequestConfig::new(model.model_name());
        Self {
            model,
            categories,
            default_class: Self::DEFAULT_CLASS.to_owned(),
            request,
            prompt: INTENT_DETECTION_PROMPT.to_owned(),
        }
    }
    /// Replaces the class returned when nothing matches.
    #[must_use]
    pub fn with_default_class(mut self, class: impl Into<String>) -> Self {
        self.default_class = class.into();
        self
    }
    /// Replaces the system prompt.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }
    /// Returns the category list, in prompt order.
    #[must_use]
    pub fn categories(&self) -> &[String] {
        &self.categories
    }
    /// Returns the fallback class.
    #[must_use]
    pub fn default_class(&self) -> &str {
        &self.default_class
    }
    /// Parses a `{"result": n}` answer into a 0-based category index.
    ///
    /// The index is 1-based on the wire, Markdown code fences are tolerated,
    /// and an out-of-range index, a non-numeric value, or malformed JSON all
    /// yield `None`.
    #[must_use]
    pub fn parse(&self, answer: &str) -> Option<usize> {
        let candidate = extract_object(answer)?;
        let parsed: Value = serde_json::from_str(&candidate).ok()?;
        let raw = parsed.get(RESULT_KEY)?;
        let one_based = match raw {
            Value::Number(number) => number.as_u64()?,
            Value::String(text) => text.trim().parse::<u64>().ok()?,
            _ => return None,
        };
        let index = usize::try_from(one_based).ok()?.checked_sub(1)?;
        (index < self.categories.len()).then_some(index)
    }
    /// Returns the category name for `answer`, or the default class.
    #[must_use]
    pub fn classify(&self, answer: &str) -> String {
        self.parse(answer)
            .and_then(|index| self.categories.get(index).cloned())
            .unwrap_or_else(|| self.default_class.clone())
    }
    /// Asks the model to classify `query` and returns the category name.
    ///
    /// A model failure or an unparsable answer is logged and mapped onto the
    /// default class, so this never returns [`Err`].
    ///
    /// # Errors
    ///
    /// Reserved for parity with the Python signature: no condition currently
    /// produces an error.
    pub async fn detect(&self, query: &str) -> Result<String> {
        let messages = [
            BaseMessage::system(self.prompt.clone()),
            BaseMessage::user(query),
        ];
        match self.model.invoke(&messages, &self.request).await {
            Ok(response) => {
                let answer = response
                    .message()
                    .map(|message| message.text().into_owned())
                    .unwrap_or_default();
                if self.parse(&answer).is_none() {
                    warn!(
                        answer = %answer,
                        default = %self.default_class,
                        "intent detection answer carried no usable category index"
                    );
                }
                Ok(self.classify(&answer))
            }
            Err(error) => {
                warn!(
                    %error,
                    default = %self.default_class,
                    "intent detection model call failed; using the default class"
                );
                Ok(self.default_class.clone())
            }
        }
    }
}
/// Extracts the JSON object of a model answer, tolerating Markdown fences.
fn extract_object(answer: &str) -> Option<String> {
    let stripped = answer
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "");
    let start = stripped.find('{')?;
    let end = stripped.rfind('}')?;
    (end > start).then(|| stripped[start..=end].to_owned())
}
/// Placeholder planner (Python's `Planner`): one undefined pending task.
///
/// Python's planner is a stub with no decomposition, no dependency wiring, and
/// no task id; this port reproduces exactly that behaviour.
#[derive(Debug, Clone, Default)]
pub struct LegacyPlanner {
    /// Task type stamped on the planned task.
    task_type: String,
}
impl LegacyPlanner {
    /// Builds a planner producing `"undefined"` tasks.
    #[must_use]
    pub fn new() -> Self {
        Self {
            task_type: LegacyTask::UNDEFINED_TASK_TYPE.to_owned(),
        }
    }
    /// Replaces the task type stamped on planned tasks.
    #[must_use]
    pub fn with_task_type(mut self, task_type: impl Into<String>) -> Self {
        self.task_type = task_type.into();
        self
    }
    /// Returns the task type stamped on planned tasks.
    #[must_use]
    pub fn task_type(&self) -> &str {
        &self.task_type
    }
    /// Plans one pending task for `event`.
    ///
    /// The task carries Python's description and metadata, and its input
    /// arguments are the serialized [`EventContent`](super::schema::EventContent)
    /// object. An unset (default-constructed) planner falls back to
    /// [`LegacyTask::UNDEFINED_TASK_TYPE`].
    #[must_use]
    pub fn plan(&self, event: &LegacyEvent) -> LegacyTask {
        let task_type = if self.task_type.is_empty() {
            LegacyTask::UNDEFINED_TASK_TYPE
        } else {
            self.task_type.as_str()
        };
        let arguments = serde_json::to_value(&event.content).unwrap_or(Value::Null);
        LegacyTask::new(task_type)
            .with_description(format!("Planner task for message: {}", event.get_query()))
            .with_input(TaskInput::default().with_arguments(arguments))
            .with_metadata("original_message_id", event.id.clone())
            .with_metadata("task_source", "planner")
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::legacy::schema::{LegacyEventType, LegacyTaskStatus};
    use openjiuwen_test_support::{ScriptedError, ScriptedModel};
    fn detector(categories: &[&str]) -> IntentDetector {
        let model = Arc::new(ScriptedModel::builder().build());
        IntentDetector::new(
            model,
            categories.iter().map(|name| (*name).to_owned()).collect(),
        )
    }
    #[test]
    fn parse_accepts_plain_and_fenced_json() {
        let detector = detector(&["weather", "chat", "code"]);
        assert_eq!(detector.parse(r#"{"result": 2}"#), Some(1));
        assert_eq!(
            detector.parse("```json\n{\"result\": 3}\n```"),
            Some(2),
            "fenced JSON is tolerated"
        );
        assert_eq!(
            detector.parse("Sure!\n```\n{\"result\": 1}\n```\ndone"),
            Some(0)
        );
    }
    #[test]
    fn parse_rejects_out_of_range_and_malformed_answers() {
        let detector = detector(&["weather", "chat"]);
        assert_eq!(detector.parse(r#"{"result": 3}"#), None, "out of range");
        assert_eq!(detector.parse(r#"{"result": 0}"#), None, "zero-based");
        assert_eq!(detector.parse(r#"{"result": -1}"#), None, "negative");
        assert_eq!(detector.parse(r#"{"other": 1}"#), None, "missing key");
        assert_eq!(detector.parse("no json here"), None);
        assert_eq!(detector.parse(r#"{"result": "two"}"#), None);
        assert_eq!(detector.parse(""), None, "an empty answer has no JSON");
    }
    #[test]
    fn parse_accepts_numeric_strings() {
        let detector = detector(&["weather", "chat"]);
        assert_eq!(detector.parse(r#"{"result": "2"}"#), Some(1));
    }
    #[test]
    fn classify_falls_back_to_the_default_class() {
        let detector = detector(&["weather"]);
        assert_eq!(detector.classify(r#"{"result": 1}"#), "weather");
        assert_eq!(detector.classify("garbage"), IntentDetector::DEFAULT_CLASS);
        assert_eq!(detector.default_class(), "unknown");
        let custom = detector.with_default_class("chat");
        assert_eq!(custom.classify("garbage"), "chat");
    }
    #[test]
    fn categories_and_prompt_are_configurable() {
        let detector = detector(&["a", "b"]).with_prompt("custom prompt");
        assert_eq!(detector.categories(), ["a".to_owned(), "b".to_owned()]);
        assert_eq!(detector.prompt, "custom prompt");
        assert!(INTENT_DETECTION_PROMPT.contains("{\"result\""));
    }
    #[tokio::test]
    async fn detect_classifies_the_model_answer() {
        let model = Arc::new(
            ScriptedModel::builder()
                .respond_text(r#"{"result": 1}"#)
                .build(),
        );
        let detector =
            IntentDetector::new(model.clone(), vec!["weather".to_owned(), "chat".to_owned()]);
        assert_eq!(detector.detect("will it rain?").await.unwrap(), "weather");
        assert_eq!(model.call_count(), 1);
        let call = model.last_call().expect("one model call");
        assert_eq!(call.messages.len(), 2);
        assert!(call.messages[0].role.is_system());
        assert_eq!(call.messages[1].text(), "will it rain?");
    }
    #[tokio::test]
    async fn detect_falls_back_on_model_failure_and_bad_answers() {
        let failing = Arc::new(
            ScriptedModel::builder()
                .fail(ScriptedError::llm("model down"))
                .build(),
        );
        let detector =
            IntentDetector::new(failing, vec!["weather".to_owned()]).with_default_class("fallback");
        assert_eq!(detector.detect("hi").await.unwrap(), "fallback");
        let unusable = Arc::new(
            ScriptedModel::builder()
                .respond_text("I cannot classify that")
                .build(),
        );
        let detector = IntentDetector::new(unusable, vec!["weather".to_owned()]);
        assert_eq!(detector.detect("hi").await.unwrap(), "unknown");
    }
    #[test]
    fn planner_produces_one_pending_undefined_task() {
        let planner = LegacyPlanner::new();
        let event = LegacyEvent::user_input("please summarise the report");
        let task = planner.plan(&event);
        assert_eq!(task.task_type, LegacyTask::UNDEFINED_TASK_TYPE);
        assert_eq!(task.status, LegacyTaskStatus::Pending);
        assert!(!task.is_terminal());
        assert_eq!(
            task.description.as_deref(),
            Some("Planner task for message: please summarise the report")
        );
        assert_eq!(task.metadata["original_message_id"], event.id.clone());
        assert_eq!(task.metadata["task_source"], "planner");
        assert_eq!(task.input.arguments["query"], "please summarise the report");
        assert!(task.task_id.is_empty(), "Python's stub leaves the id empty");
        assert_eq!(task.level, 0);
    }
    #[test]
    fn planner_task_type_is_configurable() {
        let planner = LegacyPlanner::new().with_task_type("workflow");
        assert_eq!(planner.task_type(), "workflow");
        let task = planner.plan(&LegacyEvent::new(LegacyEventType::Info));
        assert_eq!(task.task_type, "workflow");
        assert_eq!(
            task.description.as_deref(),
            Some("Planner task for message: "),
            "an event without content plans an empty description tail"
        );
        let unset = LegacyPlanner::default().plan(&LegacyEvent::user_input("hi"));
        assert_eq!(unset.task_type, LegacyTask::UNDEFINED_TASK_TYPE);
    }
}
