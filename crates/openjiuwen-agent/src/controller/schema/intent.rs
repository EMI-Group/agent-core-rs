//! Recognized controller intents.
//!
//! Ports `openjiuwen/core/controller/schema/intent.py`: an [`Intent`] pairs a
//! user [`Event`] with the operation the controller should perform, and
//! [`Intent::validate`] enforces Python's per-kind required fields. Two Rust
//! extensions (`name`, `task_type`) are optional and never required on the
//! wire.
use super::event::Event;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
/// Intent kinds (`create_task`, `pause_task`, …).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntentType {
    /// Create and run a new task.
    CreateTask,
    /// Pause a working task.
    PauseTask,
    /// Resume a paused task.
    ResumeTask,
    /// Continue from a completed task.
    ContinueTask,
    /// Supply information a task is waiting for.
    SupplementTask,
    /// Cancel a working task.
    CancelTask,
    /// Modify a working task and re-run it.
    ModifyTask,
    /// Interrupt the current task and run another one.
    SwitchTask,
    /// Unrecognized intent that needs clarification.
    UnknownTask,
}
impl IntentType {
    /// Returns the Python wire string of the intent kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CreateTask => "create_task",
            Self::PauseTask => "pause_task",
            Self::ResumeTask => "resume_task",
            Self::ContinueTask => "continue_task",
            Self::SupplementTask => "supplement_task",
            Self::CancelTask => "cancel_task",
            Self::ModifyTask => "modify_task",
            Self::SwitchTask => "switch_task",
            Self::UnknownTask => "unknown_task",
        }
    }
    /// Whether the intent mutates the task set.
    ///
    /// Reserved for later controller phases; every kind reports `false` today
    /// so routing behavior stays stable.
    #[must_use]
    pub const fn is_task_mutating(self) -> bool {
        false
    }
}
/// One recognized intent.
///
/// [`Default`] pairs [`IntentType::CreateTask`] with an empty [`Event`] and
/// full confidence, which is what serde uses for missing fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Intent {
    /// Intent kind.
    pub intent_type: IntentType,
    /// The event that triggered the intent.
    pub event: Event,
    /// Id of the task the intent targets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_task_id: Option<String>,
    /// Description of the task to create.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_task_description: Option<String>,
    /// Completed tasks a continue intent builds on.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depend_task_id: Vec<String>,
    /// Information supplied for a task waiting on input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supplementary_info: Option<String>,
    /// Details of a requested task modification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modification_details: Option<String>,
    /// Recognition confidence in `0.0..=1.0` (Python default `1.0`).
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    /// Free-form intent metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
    /// Question asked when the intent is [`IntentType::UnknownTask`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clarification_prompt: Option<String>,
    /// Rust extension: a stable label for the intent, independent of its kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Rust extension: the task type a create intent should schedule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,
}
/// Default intent confidence (Python `1.0`).
fn default_confidence() -> f32 {
    1.0
}
impl Default for Intent {
    fn default() -> Self {
        Self {
            intent_type: IntentType::CreateTask,
            event: Event::default(),
            target_task_id: None,
            target_task_description: None,
            depend_task_id: Vec::new(),
            supplementary_info: None,
            modification_details: None,
            confidence: default_confidence(),
            metadata: BTreeMap::new(),
            clarification_prompt: None,
            name: None,
            task_type: None,
        }
    }
}
impl Intent {
    /// Builds an intent of `intent_type` for `event` with full confidence.
    #[must_use]
    pub fn new(intent_type: IntentType, event: Event) -> Self {
        Self {
            intent_type,
            event,
            ..Self::default()
        }
    }
    /// Builds a create-task intent with the given description.
    #[must_use]
    pub fn create_task(event: Event, description: impl Into<String>) -> Self {
        Self::new(IntentType::CreateTask, event).with_description(description)
    }
    /// Builds the fallback create-task intent for an event.
    ///
    /// The intent is named `"default"` and takes its description from the
    /// event text (empty when the event carries none).
    #[must_use]
    pub fn default_for(event: Event) -> Self {
        let description = event.text().unwrap_or_default();
        Self::new(IntentType::CreateTask, event)
            .with_description(description)
            .with_name("default")
    }
    /// Builds an unknown intent that asks the user for clarification.
    #[must_use]
    pub fn unknown(event: Event, prompt: impl Into<String>) -> Self {
        let mut intent = Self::new(IntentType::UnknownTask, event);
        intent.clarification_prompt = Some(prompt.into());
        intent
    }
    /// Returns the intent label: [`name`](Self::name) when set, else the
    /// intent kind's wire string.
    #[must_use]
    pub fn label(&self) -> &str {
        self.name
            .as_deref()
            .unwrap_or_else(|| self.intent_type.as_str())
    }
    /// Returns the target task description, when the intent carries one.
    #[must_use]
    pub fn description(&self) -> Option<&str> {
        self.target_task_description.as_deref()
    }
    /// Returns the intent metadata, which doubles as its parameters.
    #[must_use]
    pub fn parameters(&self) -> &BTreeMap<String, Value> {
        &self.metadata
    }
    /// Validates the confidence range and the per-kind required fields.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] with Python's message for the first
    /// violated rule.
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.confidence) || self.confidence.is_nan() {
            return Err(Error::validation(format!(
                "Confidence must be between 0.0 and 1.0, got {}",
                self.confidence
            )));
        }
        let blank = |value: &Option<String>| value.as_deref().is_none_or(str::is_empty);
        match self.intent_type {
            IntentType::CreateTask => {
                if blank(&self.target_task_description) {
                    return Err(Error::validation(
                        "CREATE_TASK intent requires target_task_description",
                    ));
                }
            }
            IntentType::ContinueTask => {
                if self.depend_task_id.is_empty() {
                    return Err(Error::validation(
                        "CONTINUE_TASK intent requires depend_task_id",
                    ));
                }
            }
            IntentType::SupplementTask => {
                if blank(&self.target_task_id) {
                    return Err(Error::validation(
                        "SUPPLEMENT_TASK intent requires target_task_id",
                    ));
                }
                if blank(&self.supplementary_info) {
                    return Err(Error::validation(
                        "SUPPLEMENT_TASK intent requires supplementary_info",
                    ));
                }
            }
            IntentType::ModifyTask => {
                if blank(&self.target_task_id) {
                    return Err(Error::validation(
                        "MODIFY_TASK intent requires target_task_id",
                    ));
                }
                if blank(&self.modification_details) {
                    return Err(Error::validation(
                        "MODIFY_TASK intent requires modification_details",
                    ));
                }
            }
            IntentType::PauseTask | IntentType::ResumeTask | IntentType::CancelTask => {
                if blank(&self.target_task_id) {
                    return Err(Error::validation(format!(
                        "{} intent requires target_task_id",
                        self.intent_type.as_str()
                    )));
                }
            }
            IntentType::SwitchTask => {
                if blank(&self.target_task_description) {
                    return Err(Error::validation(
                        "SWITCH_TASK intent requires target_task_description",
                    ));
                }
            }
            IntentType::UnknownTask => {
                if blank(&self.clarification_prompt) {
                    return Err(Error::validation(
                        "UNKNOWN_TASK intent requires clarification_prompt",
                    ));
                }
            }
        }
        Ok(())
    }
    /// Sets the target task id.
    #[must_use]
    pub fn with_target_task_id(mut self, id: impl Into<String>) -> Self {
        self.target_task_id = Some(id.into());
        self
    }
    /// Sets the target task description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.target_task_description = Some(description.into());
        self
    }
    /// Sets the recognition confidence.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }
    /// Sets the intent label.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    /// Sets the task type a create intent should schedule.
    #[must_use]
    pub fn with_task_type(mut self, task_type: impl Into<String>) -> Self {
        self.task_type = Some(task_type.into());
        self
    }
    /// Adds one metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn event() -> Event {
        Event::user_input("make a report")
    }
    #[test]
    fn default_for_is_labelled_default() {
        let intent = Intent::default_for(event());
        assert_eq!(intent.intent_type, IntentType::CreateTask);
        assert_eq!(intent.label(), "default");
        assert_eq!(intent.description(), Some("make a report"));
        assert!(intent.validate().is_ok());
    }
    #[test]
    fn default_matches_python_defaults() {
        let intent = Intent::default();
        assert_eq!(intent.intent_type, IntentType::CreateTask);
        assert!((intent.confidence - 1.0).abs() < f32::EPSILON);
        assert!(intent.metadata.is_empty());
    }
    #[test]
    fn create_task_requires_a_description() {
        let intent = Intent::new(IntentType::CreateTask, event());
        let error = intent.validate().unwrap_err();
        assert!(error
            .to_string()
            .contains("CREATE_TASK intent requires target_task_description"));
        let described = Intent::new(IntentType::CreateTask, event()).with_description("write it");
        assert!(described.validate().is_ok());
    }
    #[test]
    fn pause_without_target_is_invalid() {
        let intent = Intent::new(IntentType::PauseTask, event());
        let error = intent.validate().unwrap_err();
        assert!(error
            .to_string()
            .contains("pause_task intent requires target_task_id"));
        let targeted = intent.with_target_task_id("t1");
        assert!(targeted.validate().is_ok());
    }
    #[test]
    fn unknown_requires_clarification_prompt() {
        let intent = Intent::new(IntentType::UnknownTask, event());
        let error = intent.validate().unwrap_err();
        assert!(error
            .to_string()
            .contains("UNKNOWN_TASK intent requires clarification_prompt"));
        let clarified = Intent::unknown(event(), "which task?");
        assert!(clarified.validate().is_ok());
    }
    #[test]
    fn confidence_out_of_range_is_invalid() {
        let intent = Intent::create_task(event(), "x").with_confidence(1.5);
        let error = intent.validate().unwrap_err();
        assert!(error.to_string().contains("Confidence must be between"));
        assert!(Intent::create_task(event(), "x")
            .with_confidence(f32::NAN)
            .validate()
            .is_err());
    }
    #[test]
    fn supplement_and_modify_require_their_details() {
        let supplement = Intent::new(IntentType::SupplementTask, event()).with_target_task_id("t1");
        assert!(supplement.validate().is_err());
        let supplement = {
            let mut intent = supplement;
            intent.supplementary_info = Some("info".to_owned());
            intent
        };
        assert!(supplement.validate().is_ok());
        let modify = Intent::new(IntentType::ModifyTask, event()).with_target_task_id("t1");
        assert!(modify.validate().is_err());
        let continue_intent = Intent::new(IntentType::ContinueTask, event());
        assert!(continue_intent.validate().is_err());
        let switch = Intent::new(IntentType::SwitchTask, event());
        assert!(switch.validate().is_err());
        let switch = switch.with_description("new task");
        assert!(switch.validate().is_ok());
    }
    #[test]
    fn intent_round_trips_with_snake_case_kind() {
        let intent = Intent::create_task(event(), "write it")
            .with_name("named")
            .with_task_type("agent")
            .with_metadata("k", json!(1));
        let wire = serde_json::to_value(&intent).unwrap();
        assert_eq!(wire["intent_type"], json!("create_task"));
        assert_eq!(wire["name"], json!("named"));
        assert_eq!(wire["task_type"], json!("agent"));
        let parsed: Intent = serde_json::from_value(wire).unwrap();
        assert_eq!(parsed, intent);
        assert_eq!(parsed.label(), "named");
        assert_eq!(parsed.parameters().get("k"), Some(&json!(1)));
    }
}
