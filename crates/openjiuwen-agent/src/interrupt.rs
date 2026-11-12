//! Human-in-the-loop interrupt model.
//!
//! A paused agent turn is described by [`InterruptRequest`] values; the loop
//! persists an [`InterruptionState`] under [`INTERRUPTION_STATE_KEY`] and
//! resumes by matching [`InteractiveInput`] answers to the pending requests.
use crate::output::OutputFrame;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::{BaseMessage, ToolCall};
use openjiuwen_session::{InteractionKind, InteractionRequest, InteractiveInput, Session};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
/// Session state key holding the persisted [`InterruptionState`].
pub const INTERRUPTION_STATE_KEY: &str = "__react_agent_interruption__";
/// Session state key holding the host's auto-confirm decision.
pub const AUTO_CONFIRM_STATE_KEY: &str = "__interrupt_auto_confirm__";
/// Feedback recorded when the user rejects without leaving a reason.
pub const DEFAULT_REJECT_FEEDBACK: &str = "User feedback: rejected\n action";
/// What an interrupt is asking for.
///
/// The wire form is the bare string returned by [`InterruptKind::as_str`];
/// unknown strings round-trip through [`InterruptKind::Custom`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub enum InterruptKind {
    /// No declared kind; the wire value is the empty string.
    #[default]
    Undeclared,
    /// Approval to run a protected operation.
    Permission,
    /// A yes/no confirmation of a proposed action.
    Confirm,
    /// A free-form question for the user.
    AskUser,
    /// Authorization of a skill before it runs.
    SkillAuthorization,
    /// Approval of a skill evolution proposal.
    SkillEvolutionApproval,
    /// Approval of a tool call issued on behalf of a team member.
    TeamToolApproval,
    /// The client must execute a dynamically declared tool.
    ClientTool,
    /// Any other interrupt a host application defines.
    Custom(String),
}
impl InterruptKind {
    /// Returns the wire string of this kind.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Undeclared => "",
            Self::Permission => "permission",
            Self::Confirm => "confirm",
            Self::AskUser => "ask_user",
            Self::SkillAuthorization => "skill_authorization",
            Self::SkillEvolutionApproval => "skill_evolution_approval",
            Self::TeamToolApproval => "team_tool_approval",
            Self::ClientTool => "client_tool",
            Self::Custom(value) => value,
        }
    }
    /// Returns the session interaction kind mapped from this interrupt kind.
    #[must_use]
    pub fn interaction_kind(&self) -> InteractionKind {
        match self {
            Self::Undeclared | Self::Custom(_) => InteractionKind::Custom,
            Self::Permission => InteractionKind::Permission,
            Self::Confirm => InteractionKind::Confirm,
            Self::AskUser => InteractionKind::AskUser,
            Self::SkillAuthorization => InteractionKind::SkillAuthorization,
            Self::SkillEvolutionApproval => InteractionKind::SkillEvolutionApproval,
            Self::TeamToolApproval => InteractionKind::TeamToolApproval,
            Self::ClientTool => InteractionKind::ClientTool,
        }
    }
}
impl From<&str> for InterruptKind {
    fn from(value: &str) -> Self {
        match value {
            "" => Self::Undeclared,
            "permission" => Self::Permission,
            "confirm" => Self::Confirm,
            "ask_user" => Self::AskUser,
            "skill_authorization" => Self::SkillAuthorization,
            "skill_evolution_approval" => Self::SkillEvolutionApproval,
            "team_tool_approval" => Self::TeamToolApproval,
            "client_tool" => Self::ClientTool,
            other => Self::Custom(other.to_owned()),
        }
    }
}
impl From<String> for InterruptKind {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}
impl fmt::Display for InterruptKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
impl From<&InterruptKind> for InteractionKind {
    fn from(kind: &InterruptKind) -> Self {
        kind.interaction_kind()
    }
}
impl Serialize for InterruptKind {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}
impl<'de> Deserialize<'de> for InterruptKind {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(Self::from(value.as_str()))
    }
}
/// One pending interrupt request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct InterruptRequest {
    /// Request identifier, minted as `intr_<uuid hex>`.
    pub interrupt_id: String,
    /// What is being asked.
    pub kind: InterruptKind,
    /// Id of the agent or tool that raised the request.
    pub owner: String,
    /// Human-readable question.
    pub message: String,
    /// JSON schema of the accepted answer payload.
    pub payload_schema: Value,
    /// State key the host can set to auto-confirm this request.
    pub auto_confirm_key: String,
    /// Client-rendered answer options.
    pub ui_options: Vec<Value>,
    /// Free-form metadata for the host application.
    pub metadata: BTreeMap<String, Value>,
    /// Tool the request is about, when it concerns a tool call.
    pub tool_name: String,
    /// Id of the tool call the request is about.
    pub tool_call_id: String,
    /// Raw JSON text of the tool call arguments.
    pub tool_args: String,
    /// Position of the tool call in the model's call list.
    pub index: Option<u32>,
}
impl InterruptRequest {
    /// Builds a request with a freshly minted `intr_` id and an empty schema.
    #[must_use]
    pub fn new(kind: InterruptKind, owner: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            interrupt_id: format!("intr_{}", uuid::Uuid::new_v4().simple()),
            kind,
            owner: owner.into(),
            message: message.into(),
            payload_schema: serde_json::json!({}),
            ..Self::default()
        }
    }
    /// Builds a request about the tool call `call`.
    #[must_use]
    pub fn for_tool_call(
        kind: InterruptKind,
        owner: impl Into<String>,
        message: impl Into<String>,
        call: &ToolCall,
    ) -> Self {
        Self {
            tool_name: call.name.clone(),
            tool_call_id: call.id.clone().unwrap_or_default(),
            tool_args: call.arguments.clone(),
            index: call.index,
            ..Self::new(kind, owner, message)
        }
    }
    /// Sets the auto-confirm state key.
    #[must_use]
    pub fn with_auto_confirm_key(mut self, key: impl Into<String>) -> Self {
        self.auto_confirm_key = key.into();
        self
    }
    /// Sets the answer schema.
    #[must_use]
    pub fn with_payload_schema(mut self, schema: Value) -> Self {
        self.payload_schema = schema;
        self
    }
    /// Projects the request to an `__interaction__` output frame.
    #[must_use]
    pub fn to_frame(&self, index: i64) -> OutputFrame {
        OutputFrame::interaction(
            index,
            &self.interrupt_id,
            serde_json::to_value(self).unwrap_or(Value::Null),
        )
    }
    /// Projects the request to a session [`InteractionRequest`].
    ///
    /// The question carries the whole request as JSON, `component_id` is the
    /// tool call id when the request concerns a tool call, and the request
    /// metadata is copied verbatim so the host can route the answer.
    #[must_use]
    pub fn to_interaction_request(&self) -> InteractionRequest {
        let mut request = InteractionRequest::new(
            self.kind.interaction_kind(),
            serde_json::to_value(self).unwrap_or(Value::Null),
        );
        request.id.clone_from(&self.interrupt_id);
        if !self.tool_call_id.is_empty() {
            request.component_id = Some(self.tool_call_id.clone());
        }
        request.metadata = self.metadata.clone();
        request
    }
}
/// One answer to an interrupt request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct InterruptAnswer {
    /// Owner the answer is for.
    pub owner: String,
    /// Kind of the answered request.
    pub kind: InterruptKind,
    /// Answer payload.
    pub value: Value,
    /// Id of the answered request.
    pub request_id: String,
    /// Whether the answer was carried over from a previous turn.
    pub carried: bool,
}
/// Answers matched to pending interrupts.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MatchedAnswers {
    /// Answers keyed by tool call id.
    pub answers: BTreeMap<String, Value>,
    /// Interrupt ids that received no answer.
    pub unmatched: Vec<String>,
}
/// Matches the answers in `interactive` to `pending` interrupt requests.
///
/// Each `pending` item is `(tool_call_id, interrupt_id)`. An answer keyed by
/// the interrupt id wins over one keyed by the tool call id; matches are
/// returned keyed by the tool call id. Pending items with no answer are
/// collected in [`MatchedAnswers::unmatched`].
#[must_use]
pub fn match_answers(
    interactive: &InteractiveInput,
    pending: &[(String, String)],
) -> MatchedAnswers {
    let mut answers = BTreeMap::new();
    let mut unmatched = Vec::new();
    for (tool_call_id, interrupt_id) in pending {
        if let Some(value) = interactive.interrupt_answers.get(interrupt_id) {
            answers.insert(tool_call_id.clone(), value.clone());
        } else if let Some(value) = interactive.user_inputs.get(tool_call_id) {
            answers.insert(tool_call_id.clone(), value.clone());
        } else {
            unmatched.push(interrupt_id.clone());
        }
    }
    MatchedAnswers { answers, unmatched }
}
/// Builds the canonical approval answer payload.
#[must_use]
pub fn approval_value(approved: bool, feedback: &str, auto_confirm: bool) -> Value {
    serde_json::json!({
        "approved": approved,
        "feedback": feedback,
        "auto_confirm": auto_confirm,
    })
}

/// Parses an approval answer.
///
/// Accepts a JSON boolean, an object with a boolean `approved` field, or a
/// case-insensitive approval/rejection word (`YES`, `allow`, `allow_once`,
/// `allow_always`, `no`, `deny`, `rejected`, `0`, …). Returns `None` for
/// anything else.
#[must_use]
pub fn parse_approval(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(approved) => Some(*approved),
        Value::Object(map) => map.get("approved").and_then(Value::as_bool),
        Value::String(text) => match normalized_word(text).as_str() {
            "y" | "yes" | "ok" | "approve" | "approved" | "true" | "1" | "allow" | "allow_once"
            | "a" | "always" | "allow_always" => Some(true),
            "n" | "no" | "deny" | "reject" | "rejected" | "false" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// Returns `true` when `value` grants a standing approval that covers later
/// calls.
///
/// Standing approvals are the `a`/`always`/`allow_always` answer words and
/// object payloads carrying `auto_confirm` or `persist_allow` set to `true`.
#[must_use]
pub fn is_always_approval(value: &Value) -> bool {
    match value {
        Value::String(text) => matches!(
            normalized_word(text).as_str(),
            "a" | "always" | "allow_always"
        ),
        Value::Object(map) => {
            map.get("auto_confirm").and_then(Value::as_bool) == Some(true)
                || map.get("persist_allow").and_then(Value::as_bool) == Some(true)
        }
        _ => false,
    }
}

/// Returns the human-readable text carried by an answer value.
///
/// Strings are returned as-is; `{"feedback": s}` and `{"text": s}` yield `s`;
/// `{"value": v}` recurses into `v`; anything else renders as compact JSON,
/// and `null` renders as the empty string.
#[must_use]
pub fn answer_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        Value::Object(map) => {
            if let Some(feedback) = map.get("feedback").and_then(Value::as_str) {
                feedback.to_owned()
            } else if let Some(text) = map.get("text").and_then(Value::as_str) {
                text.to_owned()
            } else if let Some(inner) = map.get("value") {
                answer_text(inner)
            } else {
                serde_json::to_string(value).unwrap_or_default()
            }
        }
        _ => serde_json::to_string(value).unwrap_or_default(),
    }
}

/// Returns the rejection feedback carried by `value`, or
/// [`DEFAULT_REJECT_FEEDBACK`] when it carries no non-empty text.
///
/// A JSON string yields itself; an object yields its `feedback`, then its
/// `reason`/`text`, then recurses into `value`. Anything else (including
/// `null`, `{}` and `{"approved": false}`) falls back to the default.
#[must_use]
pub fn reject_feedback(value: &Value) -> String {
    carried_feedback(value).unwrap_or_else(|| DEFAULT_REJECT_FEEDBACK.to_owned())
}

/// Returns the first non-empty text carried by `value`, if any.
fn carried_feedback(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => non_empty_text(text),
        Value::Object(map) => {
            if let Some(feedback) = map.get("feedback").and_then(Value::as_str) {
                if let Some(feedback) = non_empty_text(feedback) {
                    return Some(feedback);
                }
            }
            let reason = ["reason", "text"]
                .iter()
                .filter_map(|key| map.get(*key).and_then(Value::as_str))
                .find_map(non_empty_text);
            if reason.is_some() {
                return reason;
            }
            map.get("value").and_then(carried_feedback)
        }
        _ => None,
    }
}

/// Returns `text` as owned when it holds non-whitespace characters.
fn non_empty_text(text: &str) -> Option<String> {
    if text.trim().is_empty() {
        None
    } else {
        Some(text.to_owned())
    }
}

/// Lowercases `text` and trims surrounding whitespace.
fn normalized_word(text: &str) -> String {
    text.trim().to_lowercase()
}
/// Persisted state of a paused agent turn.
///
/// `requests` and `answers` are keyed by tool call id.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct InterruptionState {
    /// Loop iteration that raised the interrupts.
    pub iteration: usize,
    /// The user query that started the turn.
    pub original_query: String,
    /// Assistant message that issued the interrupting tool calls.
    pub assistant_message: Option<BaseMessage>,
    /// Pending requests keyed by tool call id.
    pub requests: BTreeMap<String, InterruptRequest>,
    /// Answers already received, keyed by tool call id.
    pub answers: BTreeMap<String, Value>,
}
impl InterruptionState {
    /// Loads the persisted state from `session`, if any.
    ///
    /// The state lives under [`INTERRUPTION_STATE_KEY`]; a missing key or an
    /// explicit `null` means there is no pending interruption.
    ///
    /// # Errors
    ///
    /// Returns an error when the session state cannot be read or the stored
    /// value is not an [`InterruptionState`].
    pub async fn load(session: &dyn Session) -> Result<Option<Self>> {
        match session.get_state(INTERRUPTION_STATE_KEY).await? {
            None | Some(Value::Null) => Ok(None),
            Some(value) => Ok(Some(serde_json::from_value(value)?)),
        }
    }
    /// Alias of [`InterruptionState::load`].
    ///
    /// # Errors
    ///
    /// Same as [`InterruptionState::load`].
    pub async fn from_session(session: &dyn Session) -> Result<Option<Self>> {
        Self::load(session).await
    }
    /// Persists the state to `session`.
    ///
    /// # Errors
    ///
    /// Returns an error when the state cannot be serialized or the session
    /// update fails.
    pub async fn save(&self, session: &dyn Session) -> Result<()> {
        let value = serde_json::to_value(self)?;
        session
            .update_state(serde_json::json!({ INTERRUPTION_STATE_KEY: value }))
            .await
    }
    /// Removes the persisted state from `session`.
    ///
    /// # Errors
    ///
    /// Returns an error when the session update fails.
    pub async fn clear(session: &dyn Session) -> Result<()> {
        session
            .update_state(serde_json::json!({ INTERRUPTION_STATE_KEY: null }))
            .await
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn interrupt_kind_wire_strings() {
        assert_eq!(InterruptKind::Undeclared.as_str(), "");
        assert_eq!(InterruptKind::Permission.as_str(), "permission");
        assert_eq!(InterruptKind::Confirm.as_str(), "confirm");
        assert_eq!(InterruptKind::AskUser.as_str(), "ask_user");
        assert_eq!(
            InterruptKind::SkillAuthorization.as_str(),
            "skill_authorization"
        );
        assert_eq!(
            InterruptKind::SkillEvolutionApproval.as_str(),
            "skill_evolution_approval"
        );
        assert_eq!(
            InterruptKind::TeamToolApproval.as_str(),
            "team_tool_approval"
        );
        assert_eq!(InterruptKind::ClientTool.as_str(), "client_tool");
        assert_eq!(InterruptKind::Custom("mine".to_owned()).as_str(), "mine");
        assert_eq!(InterruptKind::from("confirm"), InterruptKind::Confirm);
        assert_eq!(
            InterruptKind::from("mine".to_owned()),
            InterruptKind::Custom("mine".to_owned())
        );
        assert_eq!(InterruptKind::Confirm.to_string(), "confirm");
    }
    #[test]
    fn interrupt_kind_json_round_trip() {
        for kind in [
            InterruptKind::Undeclared,
            InterruptKind::Permission,
            InterruptKind::AskUser,
            InterruptKind::Custom("mine".to_owned()),
        ] {
            let value = serde_json::to_value(&kind).unwrap();
            let restored: InterruptKind = serde_json::from_value(value).unwrap();
            assert_eq!(restored, kind);
        }
        assert_eq!(
            serde_json::to_value(InterruptKind::Confirm).unwrap(),
            json!("confirm")
        );
    }
    #[test]
    fn interaction_kind_mapping() {
        assert_eq!(
            InteractionKind::from(&InterruptKind::Confirm),
            InteractionKind::Confirm
        );
        assert_eq!(
            InteractionKind::from(&InterruptKind::Custom("x".to_owned())),
            InteractionKind::Custom
        );
        assert_eq!(
            InteractionKind::from(&InterruptKind::Undeclared),
            InteractionKind::Custom
        );
    }
    #[test]
    fn request_new_has_prefixed_id_and_empty_schema() {
        let request = InterruptRequest::new(InterruptKind::Confirm, "agent", "ok?");
        assert!(request.interrupt_id.starts_with("intr_"));
        assert_eq!(request.interrupt_id.len(), "intr_".len() + 32);
        assert_eq!(request.payload_schema, json!({}));
        assert_eq!(request.owner, "agent");
        assert_eq!(request.message, "ok?");
    }
    #[test]
    fn request_for_tool_call_copies_call_fields() {
        let call = ToolCall {
            id: Some("call-1".to_owned()),
            name: "shell".to_owned(),
            arguments: "{\"cmd\":\"ls\"}".to_owned(),
            index: Some(2),
            ..ToolCall::default()
        };
        let request =
            InterruptRequest::for_tool_call(InterruptKind::Permission, "agent", "run?", &call)
                .with_auto_confirm_key(AUTO_CONFIRM_STATE_KEY)
                .with_payload_schema(json!({"type": "object"}));
        assert_eq!(request.tool_name, "shell");
        assert_eq!(request.tool_call_id, "call-1");
        assert_eq!(request.tool_args, "{\"cmd\":\"ls\"}");
        assert_eq!(request.index, Some(2));
        assert_eq!(request.auto_confirm_key, AUTO_CONFIRM_STATE_KEY);
        assert_eq!(request.payload_schema, json!({"type": "object"}));
    }
    #[test]
    fn request_converts_to_frame_and_interaction_request() {
        let request = InterruptRequest::new(InterruptKind::AskUser, "agent", "which?")
            .with_payload_schema(
                json!({"type": "object", "properties": {"answer": {"type": "string"}}}),
            );
        let frame = request.to_frame(1);
        assert_eq!(frame.kind, crate::output::INTERACTION);
        assert_eq!(frame.payload["id"], json!(request.interrupt_id.as_str()));
        assert_eq!(frame.payload["value"]["kind"], json!("ask_user"));
        let interaction = request.to_interaction_request();
        assert_eq!(interaction.id, request.interrupt_id);
        assert_eq!(interaction.kind, InteractionKind::AskUser);
        assert_eq!(interaction.question["message"], json!("which?"));
        assert_eq!(interaction.component_id, None);

        let call = ToolCall {
            id: Some("call-1".to_owned()),
            name: "shell".to_owned(),
            arguments: "{}".to_owned(),
            ..ToolCall::default()
        };
        let mut request =
            InterruptRequest::for_tool_call(InterruptKind::Permission, "agent", "run?", &call);
        request.metadata.insert("source".to_owned(), json!("tests"));
        let interaction = request.to_interaction_request();
        assert_eq!(interaction.kind, InteractionKind::Permission);
        assert_eq!(interaction.component_id.as_deref(), Some("call-1"));
        assert_eq!(interaction.metadata["source"], json!("tests"));
        assert_eq!(interaction.question["tool_name"], json!("shell"));
    }
    #[test]
    fn parse_approval_variants() {
        assert_eq!(parse_approval(&json!(true)), Some(true));
        assert_eq!(parse_approval(&json!(false)), Some(false));
        assert_eq!(parse_approval(&json!({"approved": false})), Some(false));
        assert_eq!(parse_approval(&json!({"approved": true})), Some(true));
        for word in [
            "YES",
            " Allow ",
            "y",
            "ok",
            "approve",
            "approved",
            "true",
            "1",
            "allow_once",
            "a",
            "always",
            "allow_always",
        ] {
            assert_eq!(parse_approval(&json!(word)), Some(true), "word: {word}");
        }
        for word in ["no", "Deny", "n", "reject", "rejected", "false", "0"] {
            assert_eq!(parse_approval(&json!(word)), Some(false), "word: {word}");
        }
        assert_eq!(parse_approval(&json!("maybe")), None);
        assert_eq!(parse_approval(&json!("cancel")), None);
        assert_eq!(parse_approval(&json!({"approved": "yes"})), None);
        assert_eq!(parse_approval(&json!(7)), None);
        assert_eq!(parse_approval(&Value::Null), None);
    }
    #[test]
    fn always_approval_matrix() {
        assert!(is_always_approval(&json!("A")));
        assert!(is_always_approval(&json!(" always ")));
        assert!(is_always_approval(&json!("allow_always")));
        assert!(is_always_approval(&json!({"auto_confirm": true})));
        assert!(is_always_approval(&json!({"persist_allow": true})));
        assert!(!is_always_approval(&json!("yes")));
        assert!(!is_always_approval(&json!({"auto_confirm": false})));
        assert!(!is_always_approval(&json!(true)));
        assert!(!is_always_approval(&Value::Null));
    }
    #[test]
    fn answer_text_forms() {
        assert_eq!(answer_text(&json!("plain")), "plain");
        assert_eq!(answer_text(&json!({"feedback": "nope"})), "nope");
        assert_eq!(answer_text(&json!({"text": "hello"})), "hello");
        assert_eq!(answer_text(&json!({"value": "wrapped"})), "wrapped");
        assert_eq!(answer_text(&json!({"value": {"feedback": "deep"}})), "deep");
        assert_eq!(answer_text(&json!({"other": 1})), r#"{"other":1}"#);
        assert_eq!(answer_text(&json!(3)), "3");
        assert_eq!(answer_text(&Value::Null), "");
    }
    #[test]
    fn approval_value_shape() {
        assert_eq!(
            approval_value(true, "ok", false),
            json!({
                "approved": true,
                "feedback": "ok",
                "auto_confirm": false
            })
        );
    }
    #[test]
    fn reject_feedback_prefers_reason() {
        assert_eq!(reject_feedback(&json!({"feedback": "nope"})), "nope");
        assert_eq!(reject_feedback(&json!({"reason": "bad"})), "bad");
        assert_eq!(reject_feedback(&json!({"text": "nah"})), "nah");
        assert_eq!(
            reject_feedback(&json!({"feedback": ""})),
            DEFAULT_REJECT_FEEDBACK
        );
        assert_eq!(reject_feedback(&json!({})), DEFAULT_REJECT_FEEDBACK);
        assert_eq!(reject_feedback(&json!("text")), "text");
        assert_eq!(
            reject_feedback(&json!({"other": 1})),
            DEFAULT_REJECT_FEEDBACK
        );
        assert_eq!(reject_feedback(&Value::Null), DEFAULT_REJECT_FEEDBACK);
    }
    #[test]
    fn reject_feedback_cases() {
        // (1) JSON string: used when non-empty after trim.
        assert_eq!(reject_feedback(&json!("  spaced  ")), "  spaced  ");
        assert_eq!(
            reject_feedback(&json!("   ")),
            DEFAULT_REJECT_FEEDBACK,
            "blank strings fall back"
        );
        assert_eq!(reject_feedback(&json!("")), DEFAULT_REJECT_FEEDBACK);

        // (2) `feedback` wins over `reason`/`text`, blank feedback does not.
        assert_eq!(
            reject_feedback(&json!({"feedback": "first", "reason": "second"})),
            "first"
        );
        assert_eq!(
            reject_feedback(&json!({"feedback": "", "reason": "second"})),
            "second"
        );
        assert_eq!(
            reject_feedback(&json!({"feedback": "  ", "text": "third"})),
            "third"
        );
        assert_eq!(
            reject_feedback(&json!({"feedback": "   ", "reason": ""})),
            DEFAULT_REJECT_FEEDBACK
        );

        // (3) `reason` before `text`; non-string members are ignored.
        assert_eq!(
            reject_feedback(&json!({"reason": "why", "text": "what"})),
            "why"
        );
        assert_eq!(
            reject_feedback(&json!({"feedback": 7})),
            DEFAULT_REJECT_FEEDBACK
        );

        // (4) `value` recurses.
        assert_eq!(reject_feedback(&json!({"value": "inner"})), "inner");
        assert_eq!(
            reject_feedback(&json!({"value": {"feedback": "deep"}})),
            "deep"
        );
        assert_eq!(
            reject_feedback(&json!({"value": {"value": "deeper"}})),
            "deeper"
        );
        assert_eq!(
            reject_feedback(&json!({"value": {}})),
            DEFAULT_REJECT_FEEDBACK
        );

        // (5) Anything else falls back to the default.
        assert_eq!(
            reject_feedback(&json!({"approved": false})),
            DEFAULT_REJECT_FEEDBACK
        );
        assert_eq!(
            reject_feedback(&json!({"approved": true})),
            DEFAULT_REJECT_FEEDBACK
        );
        assert_eq!(reject_feedback(&json!([1, 2])), DEFAULT_REJECT_FEEDBACK);
        assert_eq!(reject_feedback(&json!(false)), DEFAULT_REJECT_FEEDBACK);
        assert_eq!(reject_feedback(&json!(0)), DEFAULT_REJECT_FEEDBACK);
        assert_eq!(
            reject_feedback(&approval_value(false, "", false)),
            DEFAULT_REJECT_FEEDBACK
        );
        assert_eq!(
            reject_feedback(&approval_value(false, "no way", false)),
            "no way"
        );
    }
    #[test]
    fn match_answers_prefers_interrupt_id() {
        let mut interactive = InteractiveInput::default();
        interactive
            .interrupt_answers
            .insert("intr_1".to_owned(), json!({"approved": true}));
        interactive
            .user_inputs
            .insert("call-1".to_owned(), json!("lo"));
        interactive
            .user_inputs
            .insert("call-2".to_owned(), json!("hi"));
        interactive
            .user_inputs
            .insert("call-extra".to_owned(), json!("ignored"));
        interactive
            .interrupt_answers
            .insert("intr-extra".to_owned(), json!("ignored"));
        let pending = vec![
            ("call-1".to_owned(), "intr_1".to_owned()),
            ("call-2".to_owned(), "intr_2".to_owned()),
            ("call-3".to_owned(), "intr_3".to_owned()),
        ];
        let matched = match_answers(&interactive, &pending);
        assert_eq!(matched.answers.len(), 2);
        assert_eq!(matched.answers["call-1"], json!({"approved": true}));
        assert_eq!(matched.answers["call-2"], json!("hi"));
        assert!(!matched.answers.contains_key("call-3"));
        assert!(!matched.answers.contains_key("call-extra"));
        assert_eq!(matched.unmatched, vec!["intr_3".to_owned()]);
    }
    #[test]
    fn interruption_state_round_trips_through_json() {
        let request = InterruptRequest::new(InterruptKind::Confirm, "agent", "ok?");
        let state = InterruptionState {
            iteration: 2,
            original_query: "do it".to_owned(),
            requests: BTreeMap::from([("call-1".to_owned(), request)]),
            answers: BTreeMap::from([("call-1".to_owned(), json!(true))]),
            ..InterruptionState::default()
        };
        let value = serde_json::to_value(&state).unwrap();
        let restored: InterruptionState = serde_json::from_value(value).unwrap();
        assert_eq!(restored, state);
    }
    #[tokio::test]
    async fn interruption_state_save_load_clear() {
        use openjiuwen_session::AgentSession;
        let session = AgentSession::new("s1");
        assert_eq!(InterruptionState::load(&session).await.unwrap(), None);
        let state = InterruptionState {
            iteration: 1,
            original_query: "q".to_owned(),
            ..InterruptionState::default()
        };
        state.save(&session).await.unwrap();
        assert_eq!(
            InterruptionState::load(&session).await.unwrap(),
            Some(state.clone())
        );
        assert_eq!(
            InterruptionState::from_session(&session).await.unwrap(),
            Some(state)
        );
        InterruptionState::clear(&session).await.unwrap();
        assert_eq!(InterruptionState::load(&session).await.unwrap(), None);
        assert_eq!(
            session.get_state(INTERRUPTION_STATE_KEY).await.unwrap(),
            None
        );
    }
}
