//! The workflow intention detector.
//!
//! Ports `openjiuwen.dev_tools.agent_builder.builders.workflow.intention_detector`:
//! one model round trip that answers whether the dialogue already describes a
//! process ([`IntentionDetector::detect_initial`]) and whether the latest request
//! asks to modify an existing workflow ([`IntentionDetector::detect_refine`]).
//!
//! Each entry point reduces that answer to a [`BuildIntent`] and records where it
//! came from in an [`IntentionVerdict`]:
//!
//! - [`BuildIntent::Build`] — the dialogue describes a process to build from, or
//!   the request does not ask to change the existing diagram;
//! - [`BuildIntent::Clarify`] — no process description was recognised, so the
//!   builder should ask for one;
//! - [`BuildIntent::Refine`] — the request modifies the workflow the session
//!   already holds.
//!
//! # Two paths
//!
//! [`IntentionDetector::with_model`] asks a [`Model`] once and parses a fenced
//! JSON answer (`{"provide_process": bool}` or `{"need_refined": bool}`).
//! [`IntentionDetector::new`] is offline: the initial verdict is `Build` exactly
//! when the request (the retained user turns plus the current query) sequences at
//! least two steps, numbers at least two steps, or names a workflow; the refine
//! verdict is `Refine` exactly when a non-blank diagram exists and the query
//! carries a modification keyword.
//! A model failure, a missing choice, a missing fence, or an answer that does not
//! match the contract falls back to that deterministic verdict with a
//! [`GenerationSource::Fallback`] record, so detection never errors.
//!
//! A blank query is [`Error::Validation`]
//! before any model call.

use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::BaseMessage;
use serde::{Deserialize, Serialize};

use super::history::{DialogueMessage, HistoryManager};
use super::outcome::GenerationSource;
use super::utils::format_dialog_history;
use crate::util::extract_json_fence;

/// Sequencing phrases that announce a new step; the English ones match
/// case-insensitively.
///
/// This is the designer's sequence vocabulary, repeated here because the
/// detector runs before any designer stage.
const SEQUENCE_MARKERS: [&str; 10] = [
    ", then ",
    " then ",
    "after that",
    "next,",
    "首先",
    "然后",
    "接着",
    "最后",
    "，然后",
    "，再",
];

/// Words that name a process explicitly.
const WORKFLOW_KEYWORDS: [&str; 6] = ["workflow", "steps", "pipeline", "流程", "步骤", "工作流"];

/// Words that ask for a change to an existing artifact.
const MODIFICATION_KEYWORDS: [&str; 14] = [
    "modify", "change", "add", "remove", "refine", "adjust", "update", "fix", "修改", "改成",
    "增加", "删除", "调整", "优化",
];

/// The instruction sent to the model when an initial intention is detected.
const INITIAL_SYSTEM_PROMPT: &str = r"You are the openJiuwen workflow intention detector.
Decide whether the user's latest input provides a concrete process description.
Answer with exactly one ```json fenced code block and nothing else inside it,
using these keys:
- `provide_process`: true when the latest input contains concrete task steps,
  an operation sequence, or an execution order; false when it is a question, a
  vague statement, or a concept without steps
Do not add other keys and do not wrap the document in prose.";

/// The instruction sent to the model when a refinement intention is detected.
const REFINE_SYSTEM_PROMPT: &str = r"You are the openJiuwen workflow intention detector.
Decide whether the user's latest input asks to change the current Mermaid
workflow diagram.
Answer with exactly one ```json fenced code block and nothing else inside it,
using these keys:
- `need_refined`: true when the latest input adds, removes, or changes a step,
  reports an error, or otherwise differs from the current diagram; false for a
  plain confirmation or an input that asks for nothing new
Do not add other keys and do not wrap the document in prose.";

/// The decision one detection call reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildIntent {
    /// The dialogue describes a process to build from.
    #[default]
    Build,
    /// The request modifies the workflow the session already holds.
    Refine,
    /// The dialogue does not describe a process yet.
    Clarify,
}

impl BuildIntent {
    /// The intent's wire name: `"build"`, `"refine"`, or `"clarify"`.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Build => "build",
            Self::Refine => "refine",
            Self::Clarify => "clarify",
        }
    }
}

impl fmt::Display for BuildIntent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One detection verdict, with the provenance of the path that produced it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct IntentionVerdict {
    /// The decision.
    pub intent: BuildIntent,
    /// Where the decision came from.
    pub source: GenerationSource,
    /// A one-line note explaining the decision or naming a fallback reason.
    pub detail: String,
}

/// Which detection the model is asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Detection {
    /// Does the dialogue describe a process?
    Initial,
    /// Does the request modify the existing diagram?
    Refine,
}

/// The answer contract of [`Detection::Initial`].
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct InitialAnswer {
    /// Whether the latest input describes a process.
    provide_process: bool,
}

/// The answer contract of [`Detection::Refine`].
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct RefineAnswer {
    /// Whether the latest input asks for a change.
    need_refined: bool,
}

/// Detects build and refinement intentions, model-assisted when configured.
///
/// The detector owns no session state: every call decides one request from the
/// query, the caller's [`HistoryManager`], and (for refinement) the current
/// Mermaid diagram. It is [`Clone`], so a caller can move a copy into a stream.
#[derive(Clone)]
pub struct IntentionDetector {
    /// The model asked for a verdict, or `None` for the deterministic rules.
    model: Option<Arc<dyn Model>>,
}

impl IntentionDetector {
    /// Creates a detector that always uses the deterministic rules.
    #[must_use]
    pub fn new() -> Self {
        Self { model: None }
    }

    /// Creates a detector that asks `model` first and falls back to the
    /// deterministic rules whenever the answer is unusable.
    #[must_use]
    pub fn with_model(model: Arc<dyn Model>) -> Self {
        Self { model: Some(model) }
    }

    /// Detects whether the dialogue already describes a process.
    ///
    /// The deterministic rule looks at the retained user turns followed by
    /// `query`: at least two sequence markers, at least two numbered steps, or a
    /// `WORKFLOW_KEYWORDS` mention yields [`BuildIntent::Build`]; anything else
    /// yields [`BuildIntent::Clarify`].
    /// With a model configured the model is asked once, after the blank-query
    /// check; a usable answer decides the verdict, and any unusable answer falls
    /// back to the rule and records [`GenerationSource::Fallback`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `query` is empty or whitespace only.
    pub async fn detect_initial(
        &self,
        query: &str,
        history: &HistoryManager,
    ) -> Result<IntentionVerdict> {
        validate_query(query)?;
        let rule = initial_rule(query, history);
        Ok(self
            .judge(query, history, None, Detection::Initial, rule)
            .await)
    }

    /// Detects whether the request modifies `mermaid`.
    ///
    /// The deterministic rule is [`BuildIntent::Refine`] exactly when `mermaid`
    /// is non-blank and the query carries one of the
    /// `MODIFICATION_KEYWORDS`; otherwise it is [`BuildIntent::Build`].
    /// The model path asks the same question with the diagram and the formatted
    /// dialogue, and falls back exactly like [`Self::detect_initial`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `query` is empty or whitespace only.
    pub async fn detect_refine(
        &self,
        query: &str,
        history: &HistoryManager,
        mermaid: &str,
    ) -> Result<IntentionVerdict> {
        validate_query(query)?;
        let rule = refine_rule(query, mermaid);
        Ok(self
            .judge(query, history, Some(mermaid), Detection::Refine, rule)
            .await)
    }

    /// Answers with the model when one is configured, the rule otherwise.
    async fn judge(
        &self,
        query: &str,
        history: &HistoryManager,
        mermaid: Option<&str>,
        detection: Detection,
        rule: IntentionVerdict,
    ) -> IntentionVerdict {
        let Some(model) = self.model.as_ref() else {
            return rule;
        };
        let user = user_message(query, history, mermaid);
        match ask_model(model, detection, user).await {
            Ok(provide) => IntentionVerdict {
                intent: decided_intent(detection, provide),
                source: GenerationSource::Model,
                detail: model_detail(detection, provide),
            },
            Err(reason) => IntentionVerdict {
                intent: rule.intent,
                source: GenerationSource::Fallback,
                detail: format!("{reason}; deterministic rule: {}", rule.detail),
            },
        }
    }
}

impl Default for IntentionDetector {
    /// Equivalent to [`IntentionDetector::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for IntentionDetector {
    /// Reports whether a model is configured.
    ///
    /// [`Model`] is not [`fmt::Debug`], so the client is never printed.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IntentionDetector")
            .field("has_model", &self.model.is_some())
            .finish()
    }
}

/// The intent a model answer maps to.
fn decided_intent(detection: Detection, provide: bool) -> BuildIntent {
    match (detection, provide) {
        (Detection::Initial, true) | (Detection::Refine, false) => BuildIntent::Build,
        (Detection::Initial, false) => BuildIntent::Clarify,
        (Detection::Refine, true) => BuildIntent::Refine,
    }
}

/// The one-line detail recorded for an accepted model answer.
fn model_detail(detection: Detection, provide: bool) -> String {
    match detection {
        Detection::Initial => format!("model answered provide_process={provide}"),
        Detection::Refine => format!("model answered need_refined={provide}"),
    }
}

/// Rejects a blank query before any model call.
fn validate_query(query: &str) -> Result<()> {
    if query.trim().is_empty() {
        return Err(Error::validation("query must not be blank"));
    }
    Ok(())
}

/// The deterministic initial verdict over the query and retained user turns.
fn initial_rule(query: &str, history: &HistoryManager) -> IntentionVerdict {
    let context = dialogue_context(query, history);
    let lower = context.to_lowercase();
    let sequences = count_sequence_markers(&lower);
    if sequences >= 2 {
        return verdict(
            BuildIntent::Build,
            format!("the request sequences {sequences} steps"),
        );
    }
    let numbered = count_numbered_steps(&lower);
    if numbered >= 2 {
        return verdict(
            BuildIntent::Build,
            format!("the request numbers {numbered} steps"),
        );
    }
    let keywords: Vec<&str> = WORKFLOW_KEYWORDS
        .iter()
        .copied()
        .filter(|keyword| lower.contains(keyword))
        .collect();
    if keywords.is_empty() {
        verdict(
            BuildIntent::Clarify,
            "the request states no process steps yet",
        )
    } else {
        verdict(
            BuildIntent::Build,
            format!("the request names a workflow ({})", keywords.join(", ")),
        )
    }
}

/// The deterministic refinement verdict over the query and the current diagram.
fn refine_rule(query: &str, mermaid: &str) -> IntentionVerdict {
    if mermaid.trim().is_empty() {
        return verdict(BuildIntent::Build, "there is no existing diagram to refine");
    }
    let lower = query.to_lowercase();
    match MODIFICATION_KEYWORDS
        .iter()
        .find(|keyword| lower.contains(*keyword))
    {
        Some(keyword) => verdict(
            BuildIntent::Refine,
            format!("the request asks to modify the workflow (`{keyword}`)"),
        ),
        None => verdict(
            BuildIntent::Build,
            "the request does not ask to modify the diagram",
        ),
    }
}

/// A deterministic verdict with no provenance yet.
fn verdict(intent: BuildIntent, detail: impl Into<String>) -> IntentionVerdict {
    IntentionVerdict {
        intent,
        source: GenerationSource::Deterministic,
        detail: detail.into(),
    }
}

/// The retained user turns followed by `query`, newest last.
fn dialogue_context(query: &str, history: &HistoryManager) -> String {
    let mut turns: Vec<&str> = history
        .get_history()
        .iter()
        .filter(|message| message.role == "user")
        .map(|message| message.content.as_str())
        .collect();
    turns.push(query);
    turns.join("\n")
}

/// Counts the sequencing phrases in `text`, left to right and non-overlapping.
fn count_sequence_markers(text: &str) -> usize {
    let mut rest = text;
    let mut count = 0usize;
    while let Some((start, length)) = find_sequence_marker(rest) {
        count += 1;
        rest = &rest[start + length..];
    }
    count
}

/// The earliest sequencing phrase in `text`, as `(offset, length)`.
///
/// Among phrases starting at the same offset the longest one wins, so
/// `", then "` is preferred over `" then "`.
fn find_sequence_marker(text: &str) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for (index, _) in text.char_indices() {
        if best.is_some_and(|(position, _)| position < index) {
            break;
        }
        for marker in SEQUENCE_MARKERS {
            let Some(length) = match_prefix_ci(&text[index..], marker) else {
                continue;
            };
            best = match best {
                Some((position, current)) if position == index && current >= length => {
                    Some((position, current))
                }
                _ => Some((index, length)),
            };
        }
    }
    best
}

/// The byte length of `prefix` when `text` starts with it, ignoring ASCII case.
fn match_prefix_ci(text: &str, prefix: &str) -> Option<usize> {
    let mut characters = text.chars();
    for wanted in prefix.chars() {
        if !characters.next()?.eq_ignore_ascii_case(&wanted) {
            return None;
        }
    }
    Some(prefix.len())
}

/// Counts numbered list items (`1.`, `2)`, `3、`) in `text`.
fn count_numbered_steps(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut count = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        let at_boundary = index == 0 || bytes[index - 1].is_ascii_whitespace();
        if at_boundary && bytes[index].is_ascii_digit() {
            let mut digits_end = index;
            while digits_end < bytes.len() && bytes[digits_end].is_ascii_digit() {
                digits_end += 1;
            }
            let rest = &text[digits_end..];
            let marked = rest.starts_with('、')
                || rest.starts_with('.') && rest[1..].starts_with(char::is_whitespace)
                || rest.starts_with(')') && rest[1..].starts_with(char::is_whitespace);
            if marked {
                count += 1;
            }
            index = digits_end;
            continue;
        }
        index += 1;
    }
    count
}

/// The user message of a detection call: the formatted dialogue, plus the
/// diagram for refinement.
fn user_message(query: &str, history: &HistoryManager, mermaid: Option<&str>) -> String {
    let mut turns = history.get_history().to_vec();
    turns.push(DialogueMessage {
        role: "user".to_string(),
        content: query.to_string(),
    });
    let dialogue = format_dialog_history(&turns);
    match mermaid {
        Some(mermaid) => format!(
            "## Current Mermaid diagram\n{mermaid}\n\n## Dialog history and latest user input\n{dialogue}"
        ),
        None => format!("## Dialog history and latest user input\n{dialogue}"),
    }
}

/// Asks the model one detection question, or reports why it was unusable.
async fn ask_model(
    model: &Arc<dyn Model>,
    detection: Detection,
    user: String,
) -> std::result::Result<bool, String> {
    let system = match detection {
        Detection::Initial => INITIAL_SYSTEM_PROMPT,
        Detection::Refine => REFINE_SYSTEM_PROMPT,
    };
    let messages = vec![BaseMessage::system(system), BaseMessage::user(user)];
    let request = ModelRequestConfig::new(model.model_name());
    let response = model
        .invoke(&messages, &request)
        .await
        .map_err(|error| format!("model call failed: {error}"))?;
    let Some(choice) = response.choices.into_iter().next() else {
        return Err("model returned no choices".to_string());
    };
    let text = choice.message.text().into_owned();
    let value = extract_json_fence(&text).ok_or_else(|| "answer had no JSON fence".to_string())?;
    match detection {
        Detection::Initial => serde_json::from_value::<InitialAnswer>(value)
            .map(|answer| answer.provide_process)
            .map_err(|error| format!("answer did not match the contract: {error}")),
        Detection::Refine => serde_json::from_value::<RefineAnswer>(value)
            .map(|answer| answer.need_refined)
            .map_err(|error| format!("answer did not match the contract: {error}")),
    }
}

#[cfg(test)]
mod tests;
