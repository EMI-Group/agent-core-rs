//! Intent recognition and intent-to-task planning.
//!
//! Ports `openjiuwen/core/controller/modules/intent_recognizer.py` (plus the
//! `intent_toolkits` schema vocabulary) and the `legacy/reasoner` planning
//! step. Python routes an event through an LLM that calls toolkit functions;
//! this port keeps the same intent vocabulary (`create_task`, `pause_task`, …)
//! but replaces the tool-call loop with a tolerant JSON contract:
//! [`LlmIntentRecognizer`] asks the model for a JSON intent list and
//! [`RuleIntentRecognizer`] offers an offline keyword recognizer. [`Planner`]
//! then turns the recognized intents into [`Task`]s.
use super::history::{HistoryEntry, HistoryRole};
use super::schema::{Event, EventType, Intent, IntentType, Task};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::{BaseMessage, Model, ModelRequestConfig};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use tracing::warn;

/// A keyword rule that maps event text onto an intent.
///
/// This is the default (offline) recognizer's building block: the first rule
/// whose keyword appears in the event text, case-insensitively, claims the
/// event. Rules live in [`RuleIntentRecognizer`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct IntentRule {
    /// Stable rule label; it becomes the intent's [`Intent::label`].
    pub name: String,
    /// Intent kind the rule produces.
    pub intent_type: IntentType,
    /// Case-insensitive substrings that trigger the rule.
    pub keywords: Vec<String>,
    /// Task type the resulting intent should schedule, when fixed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,
    /// Task description override; the event text is used when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl Default for IntentRule {
    fn default() -> Self {
        Self {
            name: String::new(),
            intent_type: IntentType::CreateTask,
            keywords: Vec::new(),
            task_type: None,
            description: None,
        }
    }
}

impl IntentRule {
    /// Builds a rule producing `intent_type` when any of `keywords` matches.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        intent_type: IntentType,
        keywords: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            name: name.into(),
            intent_type,
            keywords: keywords.into_iter().map(Into::into).collect(),
            task_type: None,
            description: None,
        }
    }

    /// Sets the task type the resulting intent schedules.
    #[must_use]
    pub fn with_task_type(mut self, task_type: impl Into<String>) -> Self {
        self.task_type = Some(task_type.into());
        self
    }

    /// Sets a fixed task description for the resulting intent.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Whether `text` contains any of the rule's non-empty keywords.
    ///
    /// Matching is case-insensitive and substring-based; a rule without
    /// keywords never matches.
    #[must_use]
    pub fn matches(&self, text: &str) -> bool {
        let haystack = text.to_lowercase();
        self.keywords.iter().any(|keyword| {
            let needle = keyword.trim();
            !needle.is_empty() && haystack.contains(&needle.to_lowercase())
        })
    }
}

/// Stable name of the rule-based recognizer.
const RULE_RECOGNIZER_NAME: &str = "rules";

/// Stable name of the LLM-backed recognizer.
const LLM_RECOGNIZER_NAME: &str = "llm";

/// Turns an event into intents.
#[async_trait]
pub trait IntentRecognizer: Send + Sync {
    /// Returns the recognizer's stable name.
    fn name(&self) -> &str;

    /// Recognizes the intents carried by `event`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the recognizer cannot
    /// inspect the event at all; user text itself never fails recognition.
    async fn recognize(&self, event: &Event) -> Result<Vec<Intent>>;

    /// Recognizes the intents carried by `event`, given the conversation so far.
    ///
    /// `history` is the controller's trimmed conversation history of the session:
    /// oldest entry first, at most
    /// [`ControllerConfig::history_limit`](super::ControllerConfig::history_limit)
    /// messages, and the current turn excluded. The default implementation
    /// ignores it — every recognizer that does not need context therefore keeps
    /// exactly the behaviour of [`IntentRecognizer::recognize`] — while
    /// [`LlmIntentRecognizer`] replays it as prior turns. The controller calls
    /// this method instead of `recognize`.
    ///
    /// # Errors
    ///
    /// Same contract as [`IntentRecognizer::recognize`].
    async fn recognize_with_history(
        &self,
        event: &Event,
        history: &[HistoryEntry],
    ) -> Result<Vec<Intent>> {
        let _ = history;
        self.recognize(event).await
    }
}

/// Default recognizer: keyword rules, or one `default` create-task intent.
#[derive(Debug, Clone, Default)]
pub struct RuleIntentRecognizer {
    rules: Vec<IntentRule>,
}

impl RuleIntentRecognizer {
    /// Builds a recognizer without rules, which always yields the `default`
    /// create-task intent.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a recognizer from an ordered rule list.
    #[must_use]
    pub fn with_rules(rules: Vec<IntentRule>) -> Self {
        Self { rules }
    }

    /// Returns the configured rules in match order.
    #[must_use]
    pub fn rules(&self) -> &[IntentRule] {
        &self.rules
    }

    /// Appends one rule.
    pub fn add_rule(&mut self, rule: IntentRule) {
        self.rules.push(rule);
    }
}

#[async_trait]
impl IntentRecognizer for RuleIntentRecognizer {
    fn name(&self) -> &str {
        RULE_RECOGNIZER_NAME
    }

    async fn recognize(&self, event: &Event) -> Result<Vec<Intent>> {
        // Only user-facing input kinds carry a request to classify.
        if !matches!(event.event_type, EventType::Input | EventType::FollowUp) {
            return Ok(Vec::new());
        }
        let text = event.text().unwrap_or_default();
        let mut intents = Vec::new();
        for rule in &self.rules {
            if !rule.matches(&text) {
                continue;
            }
            let mut intent = Intent::new(rule.intent_type, event.clone());
            intent.target_task_id = target_task_id_from(event);
            intent.target_task_description =
                rule.description.clone().or_else(|| Some(text.clone()));
            intent.name = Some(rule.name.clone());
            intent.task_type.clone_from(&rule.task_type);
            if let Err(error) = intent.validate() {
                warn!(rule = %rule.name, %error, "skipping an invalid keyword intent");
                continue;
            }
            intents.push(intent);
        }
        let mut intents = dedupe(intents);
        if intents.is_empty() {
            intents.push(Intent::default_for(event.clone()));
        }
        Ok(intents)
    }
}

/// Default system prompt of the LLM recognizer.
///
/// The model must answer with a JSON array only; [`LlmIntentRecognizer::parse`]
/// tolerates code fences and a single object anyway.
pub const LLM_INTENT_PROMPT: &str = "\
You are the intent recognizer of a task controller.
Classify the user request into task operations and answer with a JSON array only \
(no prose, no code fence).
Allowed intent_type values: create_task, pause_task, resume_task, continue_task, \
supplement_task, cancel_task, modify_task, switch_task, unknown_task.
Each array element is an object with:
\"intent_type\" (required), \"name\" (short label), \"description\" (the task description), \
\"task_type\" (optional executor kind), \"confidence\" (0.0 to 1.0), \
\"parameters\" (optional object).
Prefer create_task whenever the request asks for new work; use unknown_task only \
when the request cannot be classified.";

/// Most conversation-history entries replayed ahead of the current event.
///
/// The replay is bounded so recognition stays deterministic and cheap no matter
/// how long the session's history is: only the newest
/// `HISTORY_REPLAY_LIMIT` non-empty entries reach the model.
const HISTORY_REPLAY_LIMIT: usize = 8;

/// LLM-backed recognizer: asks the model for a JSON intent list.
///
/// Recognition never aborts a turn: a missing answer, malformed JSON, an
/// unknown intent kind, or a model failure all fall back to the `default`
/// create-task intent.
pub struct LlmIntentRecognizer {
    model: Arc<dyn Model>,
    request: ModelRequestConfig,
    max_intents: usize,
    prompt: String,
}

impl LlmIntentRecognizer {
    /// Builds a recognizer over `model` with the default prompt and an intent
    /// budget of eight.
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        let request = ModelRequestConfig::new(model.model_name());
        Self {
            model,
            request,
            max_intents: 8,
            prompt: LLM_INTENT_PROMPT.to_owned(),
        }
    }

    /// Replaces the model request configuration.
    #[must_use]
    pub fn with_request(mut self, request: ModelRequestConfig) -> Self {
        self.request = request;
        self
    }

    /// Sets the intent budget per recognition (`0` means unlimited).
    #[must_use]
    pub fn with_max_intents(mut self, max_intents: usize) -> Self {
        self.max_intents = max_intents;
        self
    }

    /// Replaces the system prompt.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    /// Returns the configured system prompt.
    #[must_use]
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// Builds the model messages of one recognition.
    ///
    /// Always the system prompt first and `text` last, with the newest
    /// [`HISTORY_REPLAY_LIMIT`] non-empty `history` entries replayed in between
    /// as prior turns (oldest first, [`HistoryRole::User`] as a user message and
    /// [`HistoryRole::Assistant`] as an assistant message).
    fn messages_for(&self, text: &str, history: &[HistoryEntry]) -> Vec<BaseMessage> {
        let replayed: Vec<&HistoryEntry> = history
            .iter()
            .filter(|entry| !entry.text.is_empty())
            .collect();
        let start = replayed.len().saturating_sub(HISTORY_REPLAY_LIMIT);
        let replayed = &replayed[start..];
        let mut messages = Vec::with_capacity(replayed.len() + 2);
        messages.push(BaseMessage::system(self.prompt.clone()));
        messages.extend(replayed.iter().map(|entry| match entry.role {
            HistoryRole::User => BaseMessage::user(entry.text.clone()),
            HistoryRole::Assistant => BaseMessage::assistant(entry.text.clone()),
        }));
        messages.push(BaseMessage::user(text));
        messages
    }

    /// Calls the model with `messages` and parses its answer for `event`.
    ///
    /// Fail-open, exactly like every other recognition path: a model failure
    /// falls back to the `default` create-task intent.
    async fn recognize_with_messages(
        &self,
        event: &Event,
        messages: &[BaseMessage],
    ) -> Result<Vec<Intent>> {
        match self.model.invoke(messages, &self.request).await {
            Ok(response) => {
                let answer = response
                    .message()
                    .map(|message| message.text().into_owned())
                    .unwrap_or_default();
                Ok(self.parse(&answer, event))
            }
            Err(error) => {
                warn!(%error, "intent recognition model call failed; using the default intent");
                Ok(vec![Intent::default_for(event.clone())])
            }
        }
    }

    /// Tolerantly parses a model answer into intents.
    ///
    /// Handles Markdown code fences, a bare object instead of an array, the
    /// `type`/`target_task_description` aliases, and out-of-range confidence.
    /// Items with an unknown intent kind or that fail validation are dropped;
    /// an empty result yields the `default` create-task intent.
    #[must_use]
    pub fn parse(&self, answer: &str, event: &Event) -> Vec<Intent> {
        let Some(candidate) = extract_candidate(answer) else {
            warn!(answer = %answer, "intent answer carried no JSON; using the default intent");
            return vec![Intent::default_for(event.clone())];
        };
        let Ok(items) = serde_json::from_str::<Vec<Value>>(&candidate) else {
            warn!(answer = %answer, "intent answer is not a JSON list; using the default intent");
            return vec![Intent::default_for(event.clone())];
        };
        let mut intents = Vec::new();
        for item in items {
            let Ok(raw) = serde_json::from_value::<RawIntent>(item) else {
                warn!(answer = %answer, "skipping an unreadable intent item");
                continue;
            };
            let Some(intent_type) = parse_intent_type(&raw.intent_type) else {
                warn!(intent_type = %raw.intent_type, "skipping an unknown intent type");
                continue;
            };
            let mut intent = Intent::new(intent_type, event.clone());
            intent.target_task_id = target_task_id_from(event);
            intent.target_task_description = raw.description.or_else(|| event.text());
            intent.name = raw.name;
            intent.task_type = raw.task_type;
            intent.confidence = raw
                .confidence
                .filter(|confidence| (0.0..=1.0).contains(confidence))
                .unwrap_or(1.0);
            if let Some(parameters) = raw.parameters {
                intent.metadata = parameters;
            }
            if let Err(error) = intent.validate() {
                warn!(%error, "skipping an invalid intent from the model");
                continue;
            }
            intents.push(intent);
        }
        let mut intents = dedupe(intents);
        if self.max_intents > 0 {
            intents.truncate(self.max_intents);
        }
        if intents.is_empty() {
            intents.push(Intent::default_for(event.clone()));
        }
        intents
    }
}

#[async_trait]
impl IntentRecognizer for LlmIntentRecognizer {
    fn name(&self) -> &str {
        LLM_RECOGNIZER_NAME
    }

    async fn recognize(&self, event: &Event) -> Result<Vec<Intent>> {
        let Some(text) = event.text().filter(|text| !text.is_empty()) else {
            return Ok(vec![Intent::default_for(event.clone())]);
        };
        let messages = self.messages_for(&text, &[]);
        self.recognize_with_messages(event, &messages).await
    }

    async fn recognize_with_history(
        &self,
        event: &Event,
        history: &[HistoryEntry],
    ) -> Result<Vec<Intent>> {
        let Some(text) = event.text().filter(|text| !text.is_empty()) else {
            return Ok(vec![Intent::default_for(event.clone())]);
        };
        let messages = self.messages_for(&text, history);
        self.recognize_with_messages(event, &messages).await
    }
}

/// Wire shape of one intent item in the model's JSON answer.
#[derive(Debug, Deserialize)]
struct RawIntent {
    #[serde(default, alias = "type")]
    intent_type: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default, alias = "target_task_description")]
    description: Option<String>,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    task_type: Option<String>,
    #[serde(default)]
    parameters: Option<BTreeMap<String, Value>>,
}

/// Maps a Python wire intent name onto an [`IntentType`].
fn parse_intent_type(raw: &str) -> Option<IntentType> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "create_task" => Some(IntentType::CreateTask),
        "pause_task" => Some(IntentType::PauseTask),
        "resume_task" => Some(IntentType::ResumeTask),
        "continue_task" => Some(IntentType::ContinueTask),
        "supplement_task" => Some(IntentType::SupplementTask),
        "cancel_task" => Some(IntentType::CancelTask),
        "modify_task" => Some(IntentType::ModifyTask),
        "switch_task" => Some(IntentType::SwitchTask),
        "unknown_task" => Some(IntentType::UnknownTask),
        _ => None,
    }
}

/// Resolves the task an event targets: its own id, then the metadata id.
fn target_task_id_from(event: &Event) -> Option<String> {
    event.task_id.clone().or_else(|| {
        event
            .metadata
            .get("task_id")
            .and_then(Value::as_str)
            .map(str::to_owned)
    })
}

/// Extracts the JSON payload of a model answer.
///
/// Strips Markdown code fences, prefers the first `[` … last `]` span, and
/// wraps a bare `{` … `}` object in an array.
fn extract_candidate(answer: &str) -> Option<String> {
    let stripped = answer
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "");
    if let (Some(start), Some(end)) = (stripped.find('['), stripped.rfind(']')) {
        if end > start {
            return Some(stripped[start..=end].to_owned());
        }
    }
    match (stripped.find('{'), stripped.rfind('}')) {
        (Some(start), Some(end)) if end > start => {
            let body = &stripped[start..=end];
            Some(format!("[{body}]"))
        }
        _ => None,
    }
}

/// Drops later intents that repeat an earlier label.
fn dedupe(intents: Vec<Intent>) -> Vec<Intent> {
    let mut kept = Vec::new();
    let mut seen = HashSet::new();
    for intent in intents {
        if seen.insert(intent.label().to_owned()) {
            kept.push(intent);
        }
    }
    kept
}

/// Turns recognized intents into schedulable tasks.
///
/// Task ids follow Python's `<session>_task_<seq>` shape and the sequence
/// continues across [`Planner::plan`] calls until reset with
/// [`Planner::set_next_seq`].
#[derive(Debug, Clone)]
pub struct Planner {
    default_task_type: String,
    max_tasks: usize,
    next_seq: u64,
}

impl Default for Planner {
    fn default() -> Self {
        Self::new()
    }
}

impl Planner {
    /// Builds a planner with the `agent` task type, five tasks per plan, and
    /// the sequence starting at one.
    #[must_use]
    pub fn new() -> Self {
        Self {
            default_task_type: "agent".to_owned(),
            max_tasks: 5,
            next_seq: 1,
        }
    }

    /// Sets the task type used by intents that carry none.
    #[must_use]
    pub fn with_default_task_type(mut self, task_type: impl Into<String>) -> Self {
        self.default_task_type = task_type.into();
        self
    }

    /// Sets the per-plan task budget (`0` means unlimited).
    #[must_use]
    pub fn with_max_tasks(mut self, max_tasks: usize) -> Self {
        self.max_tasks = max_tasks;
        self
    }

    /// Returns the fallback task type.
    #[must_use]
    pub fn default_task_type(&self) -> &str {
        &self.default_task_type
    }

    /// Returns the per-plan task budget.
    #[must_use]
    pub fn max_tasks(&self) -> usize {
        self.max_tasks
    }

    /// Returns the next sequence number the planner will mint.
    #[must_use]
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// Sets the next sequence number the planner will mint.
    pub fn set_next_seq(&mut self, next_seq: u64) {
        self.next_seq = next_seq;
    }

    /// Plans one task per valid, distinct intent.
    ///
    /// Invalid intents and repeated labels are dropped (the first occurrence
    /// wins), and the result is truncated to [`Planner::max_tasks`] when that
    /// is non-zero. Planned tasks stay `TaskStatus::Submitted`.
    #[must_use]
    pub fn plan(&mut self, session_id: &str, intents: &[Intent]) -> Vec<Task> {
        self.plan_with_task_type(session_id, intents, None)
    }

    /// Plans tasks, overriding the task type of intents that carry none.
    ///
    /// An explicit [`Intent::task_type`] always wins over `task_type`.
    #[must_use]
    pub fn plan_with_task_type(
        &mut self,
        session_id: &str,
        intents: &[Intent],
        task_type: Option<&str>,
    ) -> Vec<Task> {
        let mut planned = Vec::new();
        let mut seen = HashSet::new();
        for intent in intents {
            if let Err(error) = intent.validate() {
                warn!(label = intent.label(), %error, "skipping an invalid intent");
                continue;
            }
            if !seen.insert(intent.label().to_owned()) {
                continue;
            }
            if self.max_tasks > 0 && planned.len() >= self.max_tasks {
                break;
            }
            let seq = self.next_seq;
            self.next_seq += 1;
            let resolved_task_type = intent
                .task_type
                .clone()
                .or_else(|| task_type.map(str::to_owned))
                .unwrap_or_else(|| self.default_task_type.clone());
            let description = intent
                .description()
                .map(str::to_owned)
                .or_else(|| intent.event.text())
                .unwrap_or_default();
            let task = Task::submitted(
                session_id,
                format!("{session_id}_task_{seq}"),
                resolved_task_type,
                description,
            )
            .with_input(intent.event.clone())
            .with_created_seq(seq)
            .with_metadata("intent_type", intent.intent_type.as_str())
            .with_metadata("intent_label", intent.label())
            .with_metadata("intent_confidence", json!(intent.confidence));
            planned.push(task);
        }
        planned
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::schema::TaskStatus;
    use openjiuwen_test_support::{ScriptedError, ScriptedModel};
    use serde_json::json;

    /// A recognizer answering every call with `answer`.
    fn llm_recognizer(answer: &str) -> LlmIntentRecognizer {
        LlmIntentRecognizer::new(Arc::new(
            ScriptedModel::builder().respond_text(answer).build(),
        ))
    }

    /// A valid, labeled create-task intent over `"do it"`.
    fn intent(label: &str) -> Intent {
        Intent::create_task(Event::user_input("do it"), "do it").with_name(label)
    }

    #[test]
    fn rule_matches_ignores_case_and_empty_keywords() {
        let rule = IntentRule::new("greet", IntentType::CreateTask, ["Hello", "  "]);
        assert!(rule.matches("say HELLO there"));
        assert!(rule.matches("hello"));
        assert!(!rule.matches("goodbye"));
        assert!(
            !IntentRule::new("empty", IntentType::CreateTask, Vec::<String>::new()).matches("")
        );
        assert!(!IntentRule::new("blank", IntentType::CreateTask, ["   "]).matches("anything"));
    }

    #[test]
    fn rule_round_trips_through_json() {
        let rule = IntentRule::new("pause", IntentType::PauseTask, ["pause", "stop"])
            .with_task_type("agent")
            .with_description("stop it");
        let wire = serde_json::to_value(&rule).unwrap();
        assert_eq!(wire["intent_type"], json!("pause_task"));
        assert_eq!(wire["task_type"], json!("agent"));
        assert_eq!(wire["keywords"], json!(["pause", "stop"]));
        let parsed: IntentRule = serde_json::from_value(wire).unwrap();
        assert_eq!(parsed, rule);

        let bare = IntentRule::new("x", IntentType::CreateTask, ["x"]);
        let wire = serde_json::to_value(&bare).unwrap();
        assert!(wire.get("task_type").is_none());
        assert!(wire.get("description").is_none());
        assert_eq!(IntentRule::default().intent_type, IntentType::CreateTask);
    }

    #[tokio::test]
    async fn rules_recognizer_defaults_to_a_single_create_task() {
        let recognizer = RuleIntentRecognizer::new();
        assert_eq!(recognizer.name(), "rules");
        let intents = recognizer
            .recognize(&Event::user_input("hi"))
            .await
            .unwrap();
        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].label(), "default");
        assert_eq!(intents[0].intent_type, IntentType::CreateTask);
        assert_eq!(intents[0].description(), Some("hi"));
        assert!(intents[0].validate().is_ok());
    }

    #[tokio::test]
    async fn rules_recognizer_dedupes_labels_and_keeps_task_type() {
        let mut recognizer = RuleIntentRecognizer::new();
        recognizer.add_rule(
            IntentRule::new("alpha", IntentType::CreateTask, ["hello"]).with_task_type("agent"),
        );
        recognizer.add_rule(IntentRule::new("alpha", IntentType::CreateTask, ["world"]));
        recognizer.add_rule(
            IntentRule::new("beta", IntentType::SwitchTask, ["world"])
                .with_description("switch it"),
        );
        assert_eq!(recognizer.rules().len(), 3);
        let intents = recognizer
            .recognize(&Event::user_input("hello world"))
            .await
            .unwrap();
        assert_eq!(intents.len(), 2, "labels must be deduplicated: {intents:?}");
        assert_eq!(intents[0].label(), "alpha");
        assert_eq!(intents[0].task_type.as_deref(), Some("agent"));
        assert_eq!(intents[1].label(), "beta");
        assert_eq!(intents[1].description(), Some("switch it"));
        assert!(intents.iter().all(|intent| intent.validate().is_ok()));
    }

    #[tokio::test]
    async fn rules_recognizer_takes_the_target_from_the_event() {
        let recognizer = RuleIntentRecognizer::with_rules(vec![IntentRule::new(
            "pause",
            IntentType::PauseTask,
            ["pause"],
        )]);
        let mut event = Event::user_input("pause it");
        event.task_id = Some("t1".to_owned());
        let intents = recognizer.recognize(&event).await.unwrap();
        assert_eq!(intents[0].target_task_id.as_deref(), Some("t1"));
        assert!(intents[0].validate().is_ok());

        let mut event = Event::follow_up("pause it");
        event.metadata.insert("task_id".to_owned(), json!("t2"));
        let intents = recognizer.recognize(&event).await.unwrap();
        assert_eq!(intents[0].target_task_id.as_deref(), Some("t2"));
    }

    #[tokio::test]
    async fn rules_recognizer_ignores_non_input_events() {
        let recognizer = RuleIntentRecognizer::with_rules(vec![IntentRule::new(
            "all",
            IntentType::CreateTask,
            ["done"],
        )]);
        let event = Event::task_completion("t1", json!({"result": "done"}));
        assert_eq!(recognizer.recognize(&event).await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn llm_recognizer_parses_a_json_array() {
        let recognizer = llm_recognizer(
            r#"[{"intent_type":"create_task","name":"report","description":"write a report"},
                 {"intent_type":"switch_task","name":"switch","task_type":"workflow"}]"#,
        );
        assert_eq!(recognizer.name(), "llm");
        let event = Event::user_input("write a report");
        let intents = recognizer.recognize(&event).await.unwrap();
        assert_eq!(intents.len(), 2, "intents: {intents:?}");
        assert_eq!(intents[0].intent_type, IntentType::CreateTask);
        assert_eq!(intents[0].label(), "report");
        assert_eq!(intents[0].description(), Some("write a report"));
        assert_eq!(intents[1].intent_type, IntentType::SwitchTask);
        assert_eq!(intents[1].task_type.as_deref(), Some("workflow"));
        assert_eq!(intents[1].description(), Some("write a report"));
        assert!(intents.iter().all(|intent| intent.validate().is_ok()));
    }

    #[tokio::test]
    async fn llm_recognizer_accepts_fenced_and_single_object_answers() {
        let fenced = llm_recognizer(
            "```json\n[{\"intent_type\":\"create_task\",\"description\":\"plan the trip\"}]\n```",
        );
        let intents = fenced
            .recognize(&Event::user_input("plan the trip"))
            .await
            .unwrap();
        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].description(), Some("plan the trip"));

        let single = llm_recognizer(r#"{"intent_type": "create_task", "description": "single"}"#);
        let intents = single
            .recognize(&Event::user_input("anything"))
            .await
            .unwrap();
        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].description(), Some("single"));
    }

    #[tokio::test]
    async fn llm_recognizer_falls_back_on_garbage() {
        let event = Event::user_input("do it");
        for answer in [
            "I cannot help with that.",
            "",
            "[not json",
            "{\"intent_type\": 3}",
        ] {
            let intents = llm_recognizer(answer).recognize(&event).await.unwrap();
            assert_eq!(intents.len(), 1, "answer {answer:?}");
            assert_eq!(intents[0].label(), "default");
            assert_eq!(intents[0].description(), Some("do it"));
        }
    }

    #[tokio::test]
    async fn llm_recognizer_truncates_to_max_intents() {
        let recognizer = llm_recognizer(
            r#"[{"intent_type":"create_task","name":"a","description":"a"},
                 {"intent_type":"create_task","name":"b","description":"b"}]"#,
        )
        .with_max_intents(1);
        let intents = recognizer
            .recognize(&Event::user_input("do both"))
            .await
            .unwrap();
        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].label(), "a");
    }

    #[tokio::test]
    async fn llm_recognizer_skips_unknown_intent_types() {
        let recognizer = llm_recognizer(
            r#"[{"intent_type":"teleport_task","description":"x"},
                 {"intent_type":"create_task","name":"kept","description":"y"}]"#,
        );
        let intents = recognizer
            .recognize(&Event::user_input("hi"))
            .await
            .unwrap();
        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].label(), "kept");
        assert_eq!(intents[0].description(), Some("y"));
    }

    #[tokio::test]
    async fn llm_recognizer_survives_a_model_failure() {
        let model = ScriptedModel::builder()
            .fail(ScriptedError::llm("provider down"))
            .build();
        let recognizer = LlmIntentRecognizer::new(Arc::new(model));
        let intents = recognizer
            .recognize(&Event::user_input("do it"))
            .await
            .unwrap();
        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].label(), "default");
        assert_eq!(intents[0].description(), Some("do it"));
    }

    #[tokio::test]
    async fn llm_recognizer_skips_empty_text_without_calling_the_model() {
        let model = ScriptedModel::builder().respond_text("unused").build();
        let recognizer = LlmIntentRecognizer::new(Arc::new(model.clone()));
        let intents = recognizer
            .recognize(&Event::new(EventType::Input))
            .await
            .unwrap();
        assert_eq!(model.call_count(), 0);
        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].label(), "default");
    }

    #[tokio::test]
    async fn llm_recognizer_maps_parameters_and_clamps_confidence() {
        let recognizer = llm_recognizer(
            r#"[{"type":"create_task","description":"x","confidence":7,"parameters":{"k":1}}]"#,
        );
        let intents = recognizer
            .recognize(&Event::user_input("hi"))
            .await
            .unwrap();
        assert_eq!(intents.len(), 1);
        assert!((intents[0].confidence - 1.0).abs() < f32::EPSILON);
        assert_eq!(intents[0].parameters().get("k"), Some(&json!(1)));

        let aliased = llm_recognizer(
            r#"[{"intent_type":"create_task","target_task_description":"aliased"}]"#,
        );
        let intents = aliased.recognize(&Event::user_input("hi")).await.unwrap();
        assert_eq!(intents[0].description(), Some("aliased"));
    }

    #[tokio::test]
    async fn llm_recognizer_sends_the_prompt_and_the_event_text() {
        let model = ScriptedModel::builder()
            .respond_text(r#"[{"intent_type":"create_task","description":"x"}]"#)
            .build();
        let recognizer = LlmIntentRecognizer::new(Arc::new(model.clone())).with_prompt("p");
        let intents = recognizer
            .recognize(&Event::user_input("do it"))
            .await
            .unwrap();
        let call = model.last_call().unwrap();
        assert_eq!(call.messages[0].text(), "p");
        assert_eq!(call.messages[1].text(), "do it");
        assert_eq!(recognizer.prompt(), "p");
        assert_eq!(intents.len(), 1);
    }

    #[tokio::test]
    async fn recognizers_are_object_safe() {
        let rules: Arc<dyn IntentRecognizer> = Arc::new(RuleIntentRecognizer::new());
        assert_eq!(rules.name(), "rules");
        assert_eq!(
            rules
                .recognize(&Event::user_input("hi"))
                .await
                .unwrap()
                .len(),
            1
        );

        let llm: Arc<dyn IntentRecognizer> = Arc::new(llm_recognizer("[]"));
        assert_eq!(llm.name(), "llm");
        let intents = llm.recognize(&Event::user_input("hi")).await.unwrap();
        assert_eq!(intents[0].label(), "default");
    }

    #[test]
    fn planner_defaults_match_the_contract() {
        let planner = Planner::new();
        assert_eq!(planner.default_task_type(), "agent");
        assert_eq!(planner.max_tasks(), 5);
        assert_eq!(planner.next_seq(), 1);
        assert_eq!(Planner::default().next_seq(), 1);
    }

    #[test]
    fn planner_dedupes_labels_and_truncates() {
        let mut planner = Planner::new();
        let tasks = planner.plan("s1", &[intent("a"), intent("a"), intent("b"), intent("c")]);
        assert_eq!(tasks.len(), 3);
        assert_eq!(tasks[0].task_id, "s1_task_1");
        assert_eq!(tasks[1].task_id, "s1_task_2");
        assert_eq!(planner.next_seq(), 4);

        let mut limited = Planner::new().with_max_tasks(1);
        let tasks = limited.plan("s1", &[intent("a"), intent("b")]);
        assert_eq!(tasks.len(), 1);
        assert_eq!(limited.next_seq(), 2);
    }

    #[test]
    fn planner_overrides_task_type_only_when_unset() {
        let mut planner = Planner::new();
        let tasks = planner.plan_with_task_type(
            "s1",
            &[intent("a"), intent("b").with_task_type("agent")],
            Some("workflow"),
        );
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].task_type, "workflow");
        assert_eq!(tasks[1].task_type, "agent");

        let mut defaulted = Planner::new().with_default_task_type("custom");
        let tasks = defaulted.plan("s1", &[intent("a")]);
        assert_eq!(tasks[0].task_type, "custom");
    }

    #[test]
    fn planner_mints_ids_metadata_and_advances_the_sequence() {
        let mut planner = Planner::new();
        planner.set_next_seq(10);
        let event = Event::user_input("write it");
        let described = Intent::create_task(event.clone(), "write it")
            .with_name("write")
            .with_confidence(0.5);
        let tasks = planner.plan("s9", &[described]);
        assert_eq!(tasks.len(), 1);
        let task = &tasks[0];
        assert_eq!(task.session_id, "s9");
        assert_eq!(task.task_id, "s9_task_10");
        assert_eq!(task.created_seq, 10);
        assert_eq!(task.description.as_deref(), Some("write it"));
        assert_eq!(task.status, TaskStatus::Submitted);
        assert_eq!(task.inputs, vec![event]);
        assert_eq!(
            task.metadata.get("intent_type"),
            Some(&json!("create_task"))
        );
        assert_eq!(task.metadata.get("intent_label"), Some(&json!("write")));
        assert_eq!(task.metadata.get("intent_confidence"), Some(&json!(0.5)));
        assert_eq!(planner.next_seq(), 11);
    }

    #[test]
    fn planner_skips_invalid_intents() {
        let mut planner = Planner::new();
        let invalid = Intent::new(IntentType::CreateTask, Event::user_input("do it"));
        let tasks = planner.plan("s1", &[invalid, intent("kept")]);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].task_id, "s1_task_1");
        assert_eq!(planner.next_seq(), 2);
    }
}

#[cfg(test)]
mod history_tests {
    use super::*;
    use openjiuwen_foundation::Role;
    use openjiuwen_test_support::{ModelCall, ScriptedModel};

    /// The text of every message of `call`, in order.
    fn texts(call: &ModelCall) -> Vec<String> {
        call.messages
            .iter()
            .map(|message| message.text().into_owned())
            .collect()
    }

    /// A one-answer scripted recognizer with the prompt `"p"`.
    fn scripted(answer: &str) -> (LlmIntentRecognizer, ScriptedModel) {
        let model = ScriptedModel::builder().respond_text(answer).build();
        let recognizer = LlmIntentRecognizer::new(Arc::new(model.clone())).with_prompt("p");
        (recognizer, model)
    }

    const ANSWER: &str = r#"[{"intent_type":"create_task","description":"x"}]"#;

    #[tokio::test]
    async fn default_history_hook_matches_plain_recognition() {
        let recognizer = RuleIntentRecognizer::new();
        let event = Event::user_input("hi");
        let plain = recognizer.recognize(&event).await.unwrap();
        let with_history = recognizer
            .recognize_with_history(
                &event,
                &[
                    HistoryEntry::user("earlier"),
                    HistoryEntry::assistant("reply"),
                ],
            )
            .await
            .unwrap();
        assert_eq!(with_history, plain);
        assert_eq!(with_history.len(), 1);
        assert_eq!(with_history[0].intent_type, IntentType::CreateTask);
        assert_eq!(with_history[0].label(), "default");
    }

    #[tokio::test]
    async fn llm_recognizer_replays_history_as_prior_turns() {
        let (recognizer, model) = scripted(ANSWER);
        let history = [
            HistoryEntry::user("earlier"),
            HistoryEntry::assistant("reply").with_task_id("t1"),
        ];
        let intents = recognizer
            .recognize_with_history(&Event::user_input("current"), &history)
            .await
            .unwrap();
        let call = model.last_call().unwrap();
        assert_eq!(texts(&call), ["p", "earlier", "reply", "current"]);
        assert_eq!(call.messages[0].role, Role::System);
        assert_eq!(call.messages[1].role, Role::User);
        assert_eq!(call.messages[2].role, Role::Assistant);
        assert_eq!(call.messages[3].role, Role::User);
        assert_eq!(model.call_count(), 1);
        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].description(), Some("x"));
    }

    #[tokio::test]
    async fn llm_recognizer_bounds_the_replayed_history() {
        let (recognizer, model) = scripted(ANSWER);
        let history: Vec<HistoryEntry> = (0..HISTORY_REPLAY_LIMIT + 2)
            .map(|index| HistoryEntry::user(format!("h{index}")))
            .collect();
        recognizer
            .recognize_with_history(&Event::user_input("current"), &history)
            .await
            .unwrap();
        let call = model.last_call().unwrap();
        assert_eq!(call.messages.len(), HISTORY_REPLAY_LIMIT + 2);
        assert_eq!(call.messages[1].text(), "h2");
        assert_eq!(
            call.messages[HISTORY_REPLAY_LIMIT].text(),
            format!("h{}", HISTORY_REPLAY_LIMIT + 1)
        );
        assert_eq!(call.messages[HISTORY_REPLAY_LIMIT + 1].text(), "current");
    }

    #[tokio::test]
    async fn llm_recognizer_skips_empty_history_entries() {
        let (recognizer, model) = scripted(ANSWER);
        let history = [
            HistoryEntry::user("kept-1"),
            HistoryEntry::user(""),
            HistoryEntry::assistant(""),
            HistoryEntry::assistant("kept-2"),
        ];
        recognizer
            .recognize_with_history(&Event::user_input("now"), &history)
            .await
            .unwrap();
        let call = model.last_call().unwrap();
        assert_eq!(texts(&call), ["p", "kept-1", "kept-2", "now"]);
    }

    #[tokio::test]
    async fn llm_recognizer_skips_empty_event_text_with_history() {
        let (recognizer, model) = scripted("unused");
        let intents = recognizer
            .recognize_with_history(
                &Event::new(EventType::Input),
                &[HistoryEntry::user("earlier")],
            )
            .await
            .unwrap();
        assert_eq!(model.call_count(), 0);
        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].label(), "default");
    }
}
