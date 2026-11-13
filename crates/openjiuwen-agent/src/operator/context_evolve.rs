//! `ContextEvolveOperator` and `MetisContextEvolveOperator`: context-window
//! evolution parameters.
//!
//! Ports the preview-operator half of
//! `openjiuwen/core/operator/context_evolve_call/` onto a concrete, tunable
//! policy: the operator owns a system prompt, an [`UpdatePolicy`], and a
//! message cap. [`ContextEvolveOperator::evolve`] applies the policy to a
//! conversation and truncates it while keeping the system message and the
//! newest turns.
//!
//! Python's `MetisContextEvolveOperator` is a preview-only handle whose real
//! state lives in an ML-backed Metis store. The Rust port keeps the identity
//! and replaces the store with a deterministic summary-header injection;
//! anything beyond that returns [`openjiuwen_common::error::Error::Unsupported`].
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{BaseMessage, MessageContent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::base::{
    apply_merge, ApplyResult, Operator, ParameterCallback, PreviewableOperator, TunableKind,
    TunableSpec, UpdateValue,
};

/// Largest `max_messages` a tuned operator may request.
pub const MAX_MESSAGES: usize = 1_000;

/// How a configured system prompt is folded into the conversation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdatePolicy {
    /// Append the prompt as a trailing system message.
    #[default]
    Append,
    /// Replace the first system message (or insert one at the front).
    Replace,
    /// Concatenate the prompt onto the first system message.
    Merge,
}

impl UpdatePolicy {
    /// The snake-case wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Append => "append",
            Self::Replace => "replace",
            Self::Merge => "merge",
        }
    }

    /// Parses a wire name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for any other value.
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "append" => Ok(Self::Append),
            "replace" => Ok(Self::Replace),
            "merge" => Ok(Self::Merge),
            other => Err(Error::validation(format!(
                "unsupported context-evolve policy '{other}'"
            ))),
        }
    }
}

/// Context-window evolution parameters with a concrete, deterministic policy.
pub struct ContextEvolveOperator {
    operator_id: String,
    system_prompt: String,
    policy: UpdatePolicy,
    max_messages: usize,
    on_parameter_updated: Option<ParameterCallback>,
}

impl ContextEvolveOperator {
    /// Creates a no-op handle (empty prompt, [`UpdatePolicy::Append`], no cap).
    #[must_use]
    pub fn new(operator_id: impl Into<String>) -> Self {
        Self {
            operator_id: operator_id.into(),
            system_prompt: String::new(),
            policy: UpdatePolicy::Append,
            max_messages: 0,
            on_parameter_updated: None,
        }
    }

    /// Sets the system prompt the policy folds in.
    #[must_use]
    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = system_prompt.into();
        self
    }

    /// Sets the folding policy.
    #[must_use]
    pub const fn with_policy(mut self, policy: UpdatePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Sets the message cap; `0` means no truncation.
    #[must_use]
    pub fn with_max_messages(mut self, max_messages: usize) -> Self {
        self.max_messages = max_messages.min(MAX_MESSAGES);
        self
    }

    /// Installs the parameter callback.
    #[must_use]
    pub fn with_callback(mut self, callback: ParameterCallback) -> Self {
        self.on_parameter_updated = Some(callback);
        self
    }

    /// The configured system prompt.
    #[must_use]
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// The folding policy.
    #[must_use]
    pub const fn policy(&self) -> UpdatePolicy {
        self.policy
    }

    /// The message cap; `0` means no truncation.
    #[must_use]
    pub const fn max_messages(&self) -> usize {
        self.max_messages
    }

    /// Applies the policy to `messages` and truncates to `max_messages`.
    ///
    /// Truncation keeps the first system message and the newest non-system
    /// turns; a cap of `0` leaves the conversation untouched.
    ///
    /// # Errors
    ///
    /// The deterministic policy cannot fail today; the signature is the public
    /// seam so an ML-backed policy can.
    #[allow(clippy::unnecessary_wraps)] // public seam reserved for fallible policies
    pub fn evolve(&self, messages: &[BaseMessage]) -> Result<Vec<BaseMessage>> {
        let mut evolved = messages.to_vec();
        let prompt = self.system_prompt.trim();
        if !prompt.is_empty() {
            match self.policy {
                UpdatePolicy::Append => evolved.push(BaseMessage::system(prompt)),
                UpdatePolicy::Replace => match first_system(&evolved) {
                    Some(index) => evolved[index].content = MessageContent::Text(prompt.to_owned()),
                    None => evolved.insert(0, BaseMessage::system(prompt)),
                },
                UpdatePolicy::Merge => match first_system(&evolved) {
                    Some(index) => {
                        let merged = format!("{}\n\n{prompt}", evolved[index].text().trim_end());
                        evolved[index].content = MessageContent::Text(merged);
                    }
                    None => evolved.insert(0, BaseMessage::system(prompt)),
                },
            }
        }
        Ok(truncate_messages(evolved, self.max_messages))
    }

    /// Parses and stores one field, returning its canonical JSON value.
    fn store(&mut self, target: &str, value: &Value) -> Result<Value> {
        match target {
            "system_prompt" => {
                let prompt = value
                    .as_str()
                    .ok_or_else(|| Error::validation("tunable 'system_prompt' expects text"))?;
                prompt.clone_into(&mut self.system_prompt);
                Ok(Value::String(self.system_prompt.clone()))
            }
            "policy" => {
                let policy = value
                    .as_str()
                    .ok_or_else(|| Error::validation("tunable 'policy' expects text"))?;
                self.policy = UpdatePolicy::parse(policy)?;
                Ok(Value::String(self.policy.as_str().to_owned()))
            }
            "max_messages" => {
                let max_messages =
                    serde_json::from_value::<usize>(value.clone()).map_err(|_| {
                        Error::validation("tunable 'max_messages' expects a non-negative integer")
                    })?;
                self.max_messages = max_messages.min(MAX_MESSAGES);
                Ok(json!(self.max_messages))
            }
            other => Err(Error::validation(format!("unknown target '{other}'"))),
        }
    }

    /// Fires the callback for `target`.
    fn fire(&self, target: &str, value: &Value) {
        if let Some(callback) = &self.on_parameter_updated {
            callback(target, value);
        }
    }
}

#[async_trait]
impl Operator for ContextEvolveOperator {
    fn operator_id(&self) -> &str {
        &self.operator_id
    }

    fn get_tunables(&self) -> Vec<TunableSpec> {
        vec![
            TunableSpec::new("system_prompt", TunableKind::Text)
                .with_description("system prompt folded into the context window"),
            TunableSpec::new("policy", TunableKind::Text)
                .with_description("fold policy: append, replace, or merge"),
            TunableSpec::new("max_messages", TunableKind::Number)
                .with_description("message cap; 0 means no truncation"),
        ]
    }

    fn get_state(&self) -> Value {
        json!({
            "system_prompt": self.system_prompt,
            "policy": self.policy.as_str(),
            "max_messages": self.max_messages,
        })
    }

    fn set_parameter(&mut self, target: &str, value: Value) -> Result<ApplyResult> {
        let stored = match self.store(target, &value) {
            Ok(stored) => stored,
            Err(error) => return Ok(ApplyResult::skipped(target, error.to_string())),
        };
        self.fire(target, &stored);
        Ok(ApplyResult::applied(target))
    }

    fn load_state(&mut self, state: Value) -> Result<()> {
        let object = state
            .as_object()
            .ok_or_else(|| Error::validation("operator state must be a JSON object"))?;
        for target in ["system_prompt", "policy", "max_messages"] {
            let Some(value) = object.get(target) else {
                continue;
            };
            let stored = self.store(target, value)?;
            self.fire(target, &stored);
        }
        Ok(())
    }

    fn apply_update(&mut self, target: &str, update: UpdateValue) -> Result<ApplyResult> {
        self.preview_update(target, &update)
    }
}

#[async_trait]
impl PreviewableOperator for ContextEvolveOperator {
    fn preview_update(&self, target: &str, update: &UpdateValue) -> Result<ApplyResult> {
        preview_against(self, target, update)
    }
}

/// Header a Metis summary is injected under.
pub const METIS_SUMMARY_HEADER: &str = "[metis-summary]";

/// Deterministic Metis context-evolution handle.
///
/// Injects a summary header into the leading system message and truncates the
/// window. The ML-backed Metis store is not ported: [`Self::learn`] returns
/// [`Error::Unsupported`].
pub struct MetisContextEvolveOperator {
    operator_id: String,
    summary: String,
    max_messages: usize,
    on_parameter_updated: Option<ParameterCallback>,
}

impl MetisContextEvolveOperator {
    /// Creates a handle with no summary and no cap.
    #[must_use]
    pub fn new(operator_id: impl Into<String>) -> Self {
        Self {
            operator_id: operator_id.into(),
            summary: String::new(),
            max_messages: 0,
            on_parameter_updated: None,
        }
    }

    /// Sets the summary text.
    #[must_use]
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = summary.into();
        self
    }

    /// Sets the message cap; `0` means no truncation.
    #[must_use]
    pub fn with_max_messages(mut self, max_messages: usize) -> Self {
        self.max_messages = max_messages.min(MAX_MESSAGES);
        self
    }

    /// Installs the parameter callback.
    #[must_use]
    pub fn with_callback(mut self, callback: ParameterCallback) -> Self {
        self.on_parameter_updated = Some(callback);
        self
    }

    /// The summary text.
    #[must_use]
    pub fn summary(&self) -> &str {
        &self.summary
    }

    /// The message cap; `0` means no truncation.
    #[must_use]
    pub const fn max_messages(&self) -> usize {
        self.max_messages
    }

    /// Injects `[metis-summary]` plus the summary into the leading system
    /// message, then truncates the window.
    ///
    /// # Errors
    ///
    /// The deterministic policy cannot fail today; the signature mirrors
    /// [`ContextEvolveOperator::evolve`].
    #[allow(clippy::unnecessary_wraps)] // public seam reserved for fallible policies
    pub fn evolve(&self, messages: &[BaseMessage]) -> Result<Vec<BaseMessage>> {
        let mut evolved = messages.to_vec();
        let summary = self.summary.trim();
        if !summary.is_empty() {
            let header = format!("{METIS_SUMMARY_HEADER}\n\n{summary}");
            match first_system(&evolved) {
                Some(index) => {
                    let existing = evolved[index].text().trim_end().to_owned();
                    let content = if existing.is_empty() {
                        header
                    } else {
                        format!("{header}\n\n{existing}")
                    };
                    evolved[index].content = MessageContent::Text(content);
                }
                None => evolved.insert(0, BaseMessage::system(header)),
            }
        }
        Ok(truncate_messages(evolved, self.max_messages))
    }

    /// Runs the ML-backed Metis learner (not ported).
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`].
    pub fn learn(&self, _experiences: &[String]) -> Result<()> {
        Err(Error::unsupported(
            "Metis ML-backed context evolution is not ported; \
             MetisContextEvolveOperator only injects a deterministic summary header",
        ))
    }

    /// Parses and stores one field, returning its canonical JSON value.
    fn store(&mut self, target: &str, value: &Value) -> Result<Value> {
        match target {
            "summary" => {
                let summary = value
                    .as_str()
                    .ok_or_else(|| Error::validation("tunable 'summary' expects text"))?;
                summary.clone_into(&mut self.summary);
                Ok(Value::String(self.summary.clone()))
            }
            "max_messages" => {
                let max_messages =
                    serde_json::from_value::<usize>(value.clone()).map_err(|_| {
                        Error::validation("tunable 'max_messages' expects a non-negative integer")
                    })?;
                self.max_messages = max_messages.min(MAX_MESSAGES);
                Ok(json!(self.max_messages))
            }
            other => Err(Error::validation(format!("unknown target '{other}'"))),
        }
    }

    /// Fires the callback for `target`.
    fn fire(&self, target: &str, value: &Value) {
        if let Some(callback) = &self.on_parameter_updated {
            callback(target, value);
        }
    }
}

#[async_trait]
impl Operator for MetisContextEvolveOperator {
    fn operator_id(&self) -> &str {
        &self.operator_id
    }

    fn get_tunables(&self) -> Vec<TunableSpec> {
        vec![
            TunableSpec::new("summary", TunableKind::Text)
                .with_description("task-memory summary injected under [metis-summary]"),
            TunableSpec::new("max_messages", TunableKind::Number)
                .with_description("message cap; 0 means no truncation"),
        ]
    }

    fn get_state(&self) -> Value {
        json!({
            "summary": self.summary,
            "max_messages": self.max_messages,
        })
    }

    fn set_parameter(&mut self, target: &str, value: Value) -> Result<ApplyResult> {
        let stored = match self.store(target, &value) {
            Ok(stored) => stored,
            Err(error) => return Ok(ApplyResult::skipped(target, error.to_string())),
        };
        self.fire(target, &stored);
        Ok(ApplyResult::applied(target))
    }

    fn load_state(&mut self, state: Value) -> Result<()> {
        let object = state
            .as_object()
            .ok_or_else(|| Error::validation("operator state must be a JSON object"))?;
        for target in ["summary", "max_messages"] {
            let Some(value) = object.get(target) else {
                continue;
            };
            let stored = self.store(target, value)?;
            self.fire(target, &stored);
        }
        Ok(())
    }

    fn apply_update(&mut self, target: &str, update: UpdateValue) -> Result<ApplyResult> {
        self.preview_update(target, &update)
    }
}

#[async_trait]
impl PreviewableOperator for MetisContextEvolveOperator {
    fn preview_update(&self, target: &str, update: &UpdateValue) -> Result<ApplyResult> {
        preview_against(self, target, update)
    }
}

/// Index of the first system message.
fn first_system(messages: &[BaseMessage]) -> Option<usize> {
    messages.iter().position(|message| message.role.is_system())
}

/// Truncates `messages` to `limit`, keeping the first system message and the
/// newest non-system turns; `0` means no truncation.
fn truncate_messages(messages: Vec<BaseMessage>, limit: usize) -> Vec<BaseMessage> {
    if limit == 0 || messages.len() <= limit {
        return messages;
    }
    let Some(system_index) = first_system(&messages) else {
        return messages[messages.len() - limit..].to_vec();
    };
    let system = messages[system_index].clone();
    let mut newest = Vec::new();
    let mut budget = limit.saturating_sub(1);
    for (index, message) in messages.iter().enumerate().rev() {
        if index == system_index || message.role.is_system() {
            continue;
        }
        if budget == 0 {
            break;
        }
        newest.push(message.clone());
        budget -= 1;
    }
    let mut truncated = Vec::with_capacity(newest.len() + 1);
    truncated.push(system);
    truncated.extend(newest.into_iter().rev());
    truncated
}

/// Computes what `update` would do to `target` without mutating `operator`.
fn preview_against<T: Operator + ?Sized>(
    operator: &T,
    target: &str,
    update: &UpdateValue,
) -> Result<ApplyResult> {
    let Some(spec) = operator
        .get_tunables()
        .into_iter()
        .find(|spec| spec.name == target)
    else {
        return Ok(ApplyResult::skipped(target, "unknown target"));
    };
    let state = operator.get_state();
    let current = state.get(target).cloned().unwrap_or(Value::Null);
    let next = apply_merge(&current, update);
    spec.validate(&next)?;
    if next == current {
        return Ok(ApplyResult::skipped(target, "value unchanged"));
    }
    Ok(ApplyResult::applied(target))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use std::sync::Arc;

    fn conversation() -> Vec<BaseMessage> {
        vec![
            BaseMessage::system("old system"),
            BaseMessage::user("first"),
            BaseMessage::assistant("reply"),
            BaseMessage::user("second"),
        ]
    }

    type Recorded = Arc<parking_lot::Mutex<Vec<(String, Value)>>>;

    fn recording_callback() -> (ParameterCallback, Recorded) {
        let log = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let sink = Arc::clone(&log);
        let callback: ParameterCallback = Arc::new(move |target: &str, value: &Value| {
            sink.lock().push((target.to_owned(), value.clone()));
        });
        (callback, log)
    }

    #[test]
    fn policy_wire_names_round_trip() {
        assert_eq!(UpdatePolicy::default(), UpdatePolicy::Append);
        assert_eq!(UpdatePolicy::parse("MERGE").unwrap(), UpdatePolicy::Merge);
        assert_eq!(UpdatePolicy::Replace.as_str(), "replace");
        assert!(UpdatePolicy::parse("shrink").is_err());
        assert_eq!(
            serde_json::to_value(UpdatePolicy::Append).unwrap(),
            json!("append")
        );
    }

    #[test]
    fn append_replace_and_merge_policies() {
        let messages = conversation();

        let appended = ContextEvolveOperator::new("op")
            .with_system_prompt("tuned")
            .evolve(&messages)
            .unwrap();
        assert_eq!(appended.len(), 5);
        assert_eq!(appended[4].text(), "tuned");
        assert!(appended[4].role.is_system());

        let replaced = ContextEvolveOperator::new("op")
            .with_system_prompt("tuned")
            .with_policy(UpdatePolicy::Replace)
            .evolve(&messages)
            .unwrap();
        assert_eq!(replaced.len(), 4);
        assert_eq!(replaced[0].text(), "tuned");

        let merged = ContextEvolveOperator::new("op")
            .with_system_prompt("tuned")
            .with_policy(UpdatePolicy::Merge)
            .evolve(&messages)
            .unwrap();
        assert_eq!(merged[0].text(), "old system\n\ntuned");
        assert_eq!(merged.len(), 4);
    }

    #[test]
    fn policies_insert_a_system_message_when_missing() {
        let messages = vec![BaseMessage::user("hi")];
        let operator = ContextEvolveOperator::new("op")
            .with_system_prompt("tuned")
            .with_policy(UpdatePolicy::Replace);
        let evolved = operator.evolve(&messages).unwrap();
        assert_eq!(evolved.len(), 2);
        assert!(evolved[0].role.is_system());
        assert_eq!(evolved[0].text(), "tuned");
    }

    #[test]
    fn truncation_keeps_the_system_message_and_newest_turns() {
        let messages = conversation();
        let operator = ContextEvolveOperator::new("op").with_max_messages(2);
        let evolved = operator.evolve(&messages).unwrap();
        assert_eq!(evolved.len(), 2);
        assert_eq!(evolved[0].text(), "old system");
        assert_eq!(evolved[1].text(), "second");

        let capped_with_prompt = ContextEvolveOperator::new("op")
            .with_system_prompt("tuned")
            .with_policy(UpdatePolicy::Replace)
            .with_max_messages(3)
            .evolve(&messages)
            .unwrap();
        assert_eq!(capped_with_prompt.len(), 3);
        assert_eq!(capped_with_prompt[0].text(), "tuned");
        assert_eq!(capped_with_prompt[2].text(), "second");

        let untruncated = ContextEvolveOperator::new("op")
            .with_max_messages(0)
            .evolve(&messages)
            .unwrap();
        assert_eq!(untruncated.len(), 4);
    }

    #[test]
    fn set_parameter_and_state_round_trip() {
        let (callback, log) = recording_callback();
        let mut operator = ContextEvolveOperator::new("op").with_callback(callback);
        operator.set_parameter("policy", json!("replace")).unwrap();
        operator.set_parameter("max_messages", json!(2)).unwrap();
        operator
            .set_parameter("system_prompt", json!("tuned"))
            .unwrap();
        assert_eq!(operator.policy(), UpdatePolicy::Replace);
        assert_eq!(operator.max_messages(), 2);
        let targets: Vec<String> = log
            .lock()
            .iter()
            .map(|(target, _)| target.clone())
            .collect();
        assert_eq!(targets, ["policy", "max_messages", "system_prompt"]);
        assert!(
            !operator
                .set_parameter("policy", json!("shrink"))
                .unwrap()
                .applied
        );

        let state = operator.get_state();
        let mut restored = ContextEvolveOperator::new("op");
        restored.load_state(state.clone()).unwrap();
        assert_eq!(restored.get_state(), state);
    }

    #[test]
    fn preview_update_does_not_mutate() {
        let operator = ContextEvolveOperator::new("op").with_system_prompt("keep");
        let before = operator.get_state();
        let preview = operator
            .preview_update("system_prompt", &UpdateValue::set(json!("changed")))
            .unwrap();
        assert!(preview.applied);
        assert_eq!(operator.get_state(), before);

        let unknown = operator
            .preview_update("nope", &UpdateValue::set(json!(1)))
            .unwrap();
        assert!(!unknown.applied);
        assert!(operator
            .preview_update("max_messages", &UpdateValue::set(json!("many")))
            .is_err());

        let mut mutable = ContextEvolveOperator::new("op").with_system_prompt("keep");
        mutable
            .apply_update("system_prompt", UpdateValue::set(json!("changed")))
            .unwrap();
        assert_eq!(mutable.system_prompt(), "keep");
    }

    #[test]
    fn metis_injects_a_summary_header_and_truncates() {
        let messages = conversation();
        let operator = MetisContextEvolveOperator::new("metis")
            .with_summary("task summary")
            .with_max_messages(2);
        let evolved = operator.evolve(&messages).unwrap();
        assert_eq!(evolved.len(), 2);
        assert_eq!(
            evolved[0].text(),
            "[metis-summary]\n\ntask summary\n\nold system"
        );
        assert_eq!(evolved[1].text(), "second");
        assert!(evolved[0].role.is_system());

        assert_eq!(
            operator.get_state(),
            json!({"summary": "task summary", "max_messages": 2})
        );
        let error = operator.learn(&["experience".to_owned()]).unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
    }

    #[test]
    fn metis_preview_and_state_round_trip() {
        let mut operator = MetisContextEvolveOperator::new("metis");
        operator.set_parameter("summary", json!("s1")).unwrap();
        operator.set_parameter("max_messages", json!(4)).unwrap();
        let before = operator.get_state();
        assert!(
            operator
                .preview_update("summary", &UpdateValue::append(json!(" more")))
                .unwrap()
                .applied
        );
        assert_eq!(operator.get_state(), before);
        let state = operator.get_state();
        let mut restored = MetisContextEvolveOperator::new("metis");
        restored.load_state(state.clone()).unwrap();
        assert_eq!(restored.get_state(), state);
    }
}
