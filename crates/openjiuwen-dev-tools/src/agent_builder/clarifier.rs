//! The requirement clarifier: one decision about whether a request can be
//! designed from.
//!
//! Ports the decision half of
//! `openjiuwen/dev_tools/agent_builder/builders/llm_agent/clarifier.py`
//! (the factor-extraction stage). Python runs two model round trips — one to
//! extract the agent's basic elements and one to plan resources — and feeds the
//! result into the designer. This port reduces that to the single question the
//! builder needs answered: is the request complete enough to design from, or is
//! one specific question still missing?
//!
//! The answer is a [`Clarification`] with three fields: `ready`, the one
//! `question` to ask when it is not ready, and a `summary` of the requirements
//! recognised so far.
//!
//! # Two paths
//!
//! [`Clarifier::with_model`] asks a [`Model`] once and parses a fenced JSON
//! answer; [`Clarifier::new`] is offline and uses a deterministic keyword
//! heuristic over the dialogue. The model path is best-effort: a failed call, a
//! missing choice, a missing JSON fence, or a document that does not match the
//! contract all fall back to the heuristic, so [`Clarifier::clarify`] only ever
//! fails on a blank query.
//!
//! # The deterministic heuristic
//!
//! The heuristic lowercases the contents of every retained `"user"` history
//! turn (oldest first) followed by the current query, and looks for four
//! keyword-based signals, Chinese and English:
//!
//! - **goal** — what the agent should do (`agent`, `create`, `summarize`,
//!   `智能体`, `生成`, …);
//! - **input** — where its data comes from (`input`, `document`, `user`,
//!   `输入`, `文档`, …) or a `${...}` placeholder;
//! - **output** — what it should produce (`output`, `return`, `result`,
//!   `输出`, `返回`, …);
//! - **tools** — tools and constraints (`tool`, `api`, `search`, `must`, `工具`,
//!   `约束`, …).
//!
//! A request is `ready` exactly when the goal and the expected output are both
//! stated — those two are the minimum a design needs. When it is not ready the
//! question names the first missing piece in the order goal, input, output,
//! tools, and `summary` lists the detected signals in that same fixed order.
//!
//! # Rounds, provenance, and streaming
//!
//! [`Clarifier::clarify`] is the single-pass verdict and is implemented as a
//! one-round dialogue, so both entry points share one judgement.
//! The bounded multi-turn loop ([`Clarifier::clarify_rounds`],
//! [`Clarifier::clarify_rounds_with`]) asks the same judgement repeatedly over a
//! working copy of the dialogue and terminates on [`ClarificationTermination`];
//! it lives in [`rounds`].
//! Every verdict records a [`StageProvenance`](super::outcome::StageProvenance)
//! entry in [`Clarification::provenance`], so an accepted model answer, the
//! offline heuristic, and a fallback after a failed model call are
//! distinguishable, and a degraded verdict is logged at warn level.
//! [`Clarifier::stream_clarify`] yields the same rounds one by one.

pub mod rounds;

use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::BaseMessage;
use serde::{Deserialize, Serialize};

use super::history::HistoryManager;
use super::outcome::{GenerationProvenance, GenerationSource};
use super::progress::ProgressStage;
use crate::util::extract_json_fence;

pub use rounds::{
    ClarificationDialogue, ClarificationRound, ClarificationTermination, ClarifyConfig,
    ClarifyEvent, DEFAULT_MAX_CLARIFY_ROUNDS,
};

/// The instruction sent to the model when a clarifier has one.
///
/// It names exactly the keys the parser reads, so a compliant answer round-trips
/// into a [`Clarification`] without repair.
const CLARIFIER_SYSTEM_PROMPT: &str = r"You are the openJiuwen requirement clarifier.
Decide whether the user's request states both what the agent should do and what
it should produce.

Answer with exactly one ```json fenced code block and nothing else inside it,
using these keys:
- `ready`: true when the goal and the expected output are stated, false otherwise
- `question`: when `ready` is false, ONE question asking for the single missing
  detail; empty when `ready` is true
- `summary`: a short list of the requirements you recognised, or an empty string

Do not add other keys and do not wrap the document in prose.";

/// The question asked when the request does not state a goal.
const GOAL_QUESTION: &str = "What should the agent accomplish?";
/// The question asked when the request does not state its input.
const INPUT_QUESTION: &str = "What input will the agent receive?";
/// The question asked when the request does not state its output.
const OUTPUT_QUESTION: &str = "What should the agent produce as its output?";
/// The question asked when the request does not state tools or constraints.
const TOOLS_QUESTION: &str =
    "Does the agent need any tools or constraints (APIs, plugins, databases)?";

/// Words that signal what the agent should do.
const GOAL_KEYWORDS: &[&str] = &[
    "agent",
    "assistant",
    "bot",
    "workflow",
    "help",
    "create",
    "build",
    "assist",
    "answer",
    "summarize",
    "translate",
    "generate",
    "analyze",
    "extract",
    "classify",
    "chat",
    "客服",
    "助手",
    "智能体",
    "工作流",
    "创建",
    "生成",
    "回答",
    "帮助",
    "总结",
    "翻译",
    "分析",
    "提取",
    "分类",
];

/// Words that signal where the agent's data comes from.
const INPUT_KEYWORDS: &[&str] = &[
    "input", "from", "given", "provided", "read", "document", "text", "question", "user", "参数",
    "输入", "读取", "给定", "用户", "提供", "传入", "文本", "问题", "文档",
];

/// Words that signal what the agent should produce.
const OUTPUT_KEYWORDS: &[&str] = &[
    "output", "return", "respond", "response", "reply", "result", "answer", "export", "write",
    "输出", "返回", "回复", "结果", "回答", "导出",
];

/// Words that signal tool or constraint requirements.
const TOOLS_KEYWORDS: &[&str] = &[
    "tool",
    "api",
    "plugin",
    "search",
    "database",
    "constraint",
    "must",
    "only",
    "禁止",
    "工具",
    "插件",
    "接口",
    "搜索",
    "数据库",
    "约束",
    "限制",
    "必须",
];

/// The clarifier's verdict on one request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Clarification {
    /// Whether the request is complete enough to design from.
    pub ready: bool,
    /// The one missing detail to ask about; empty when [`Self::ready`].
    pub question: String,
    /// The detected requirement signals, in the fixed order
    /// `"goal, input, output, tools"` (empty when none was detected).
    pub summary: String,
    /// Where the verdict came from, as one [`ProgressStage::Clarifying`] record.
    pub provenance: GenerationProvenance,
}

/// The provenance detail recorded when no model is wired for a verdict.
pub(crate) const NO_MODEL_DETAIL: &str = "no model configured; keyword heuristic";
/// The provenance detail recorded when a wired model's call returns an error.
pub(crate) const MODEL_CALL_FAILED: &str = "model call failed";
/// The provenance detail recorded when a wired model answers without a choice.
pub(crate) const MODEL_NO_CHOICES: &str = "model returned no choices";
/// The provenance detail recorded when a model answer holds no JSON fence.
pub(crate) const MODEL_NO_FENCE: &str = "answer had no JSON fence";
/// The provenance detail recorded for an answer that breaks the answer contract.
pub(crate) const MODEL_CONTRACT_MISMATCH: &str = "answer did not match the contract";

/// The answer contract of the model path.
///
/// `ready` is required — an answer without it does not match the contract and
/// falls back to the heuristic — while `question` and `summary` may be omitted.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct ClarificationAnswer {
    /// Whether the request is complete enough to design from.
    ready: bool,
    /// The one missing detail to ask about, if any.
    #[serde(default)]
    question: String,
    /// The requirement signals the model recognised, if any.
    #[serde(default)]
    summary: String,
}

/// Requirement clarifier: model-assisted when a model is configured,
/// deterministic otherwise.
///
/// The clarifier owns no session state: every call decides one request from the
/// query plus the caller's [`HistoryManager`]. It is [`Clone`], so a caller can
/// move a copy into a stream.
#[derive(Clone)]
pub struct Clarifier {
    /// The model asked for a verdict, or `None` for the offline heuristic.
    model: Option<Arc<dyn Model>>,
}

impl Clarifier {
    /// Creates an offline clarifier that always uses the deterministic
    /// heuristic.
    #[must_use]
    pub fn new() -> Self {
        Self { model: None }
    }

    /// Creates a clarifier that asks `model` first and falls back to the
    /// deterministic heuristic whenever the answer is unusable.
    #[must_use]
    pub fn with_model(model: Arc<dyn Model>) -> Self {
        Self { model: Some(model) }
    }

    /// Decides whether the request can be designed from.
    ///
    /// A blank query is rejected before any model call. With a model configured
    /// the model is called once — one system message describing the contract,
    /// one message per retained history turn (`"assistant"` turns become
    /// assistant messages, every other role becomes a user message), and the
    /// query as the final user message. A parsed answer is returned as-is, with
    /// its strings trimmed.
    ///
    /// Any unusable model answer — a failed call, no choice, no JSON fence, or a
    /// document that is not an object with a boolean `ready` — is logged at
    /// debug level and falls back to the deterministic heuristic documented in
    /// the [module docs](self). Without a model, the heuristic answers directly.
    /// The verdict records which of those paths produced it in
    /// [`Clarification::provenance`].
    ///
    /// This is [`Clarifier::clarify_rounds_with`] with one round and no
    /// responder, so a single-pass verdict and the first round of a dialogue are
    /// always the same judgement.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation)
    /// when `query` is empty or whitespace only.
    pub async fn clarify(&self, query: &str, history: &HistoryManager) -> Result<Clarification> {
        let dialogue = self
            .clarify_rounds_with(query, history, &ClarifyConfig::new(1), |_| None)
            .await?;
        Ok(dialogue.clarification)
    }

    /// Judges one round: the model when one is wired, the heuristic otherwise.
    ///
    /// A wired model's unusable answer is a [`GenerationSource::Fallback`] over
    /// the heuristic, never an error; the detail names the reason.
    async fn judge(&self, query: &str, history: &HistoryManager) -> Judgement {
        if let Some(model) = self.model.as_ref() {
            return match ask_model(model, query, history).await {
                Ok(clarification) => Judgement {
                    clarification,
                    source: GenerationSource::Model,
                    detail: String::new(),
                },
                Err(reason) => Judgement {
                    clarification: deterministic(query, history),
                    source: GenerationSource::Fallback,
                    detail: reason.to_owned(),
                },
            };
        }
        Judgement {
            clarification: deterministic(query, history),
            source: GenerationSource::Deterministic,
            detail: NO_MODEL_DETAIL.to_owned(),
        }
    }
}

impl Default for Clarifier {
    /// Equivalent to [`Clarifier::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Clarifier {
    /// Reports whether a model is configured.
    ///
    /// [`Model`] is not [`fmt::Debug`], so the client is never printed.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Clarifier")
            .field("has_model", &self.model.is_some())
            .finish()
    }
}

/// One round's verdict plus where it came from.
///
/// The verdict's own provenance is empty until the caller finalizes it, because
/// a dialogue aggregates one record per round.
pub(crate) struct Judgement {
    /// The verdict itself, with an empty provenance.
    pub(crate) clarification: Clarification,
    /// Where this verdict came from.
    pub(crate) source: GenerationSource,
    /// A short note naming a fallback reason; empty for an accepted model
    /// answer.
    pub(crate) detail: String,
}

impl Judgement {
    /// This verdict as one [`ProgressStage::Clarifying`] record.
    pub(crate) fn provenance(&self) -> GenerationProvenance {
        let mut provenance = GenerationProvenance::new();
        provenance.record(ProgressStage::Clarifying, self.source, self.detail.clone());
        provenance
    }
}

/// Asks the model for a verdict, reporting the reason for an unusable answer.
async fn ask_model(
    model: &Arc<dyn Model>,
    query: &str,
    history: &HistoryManager,
) -> std::result::Result<Clarification, &'static str> {
    let messages = conversation(query, history);
    let request = ModelRequestConfig::new(model.model_name());
    let response = match model.invoke(&messages, &request).await {
        Ok(response) => response,
        Err(error) => {
            tracing::debug!(%error, "requirement clarifier model call failed");
            return Err(MODEL_CALL_FAILED);
        }
    };
    let Some(choice) = response.choices.into_iter().next() else {
        tracing::debug!("requirement clarifier model returned no choices");
        return Err(MODEL_NO_CHOICES);
    };
    parse_answer(&choice.message.text())
}

/// Assembles the system contract, the history turns, and the query.
fn conversation(query: &str, history: &HistoryManager) -> Vec<BaseMessage> {
    let mut messages = Vec::with_capacity(history.len() + 2);
    messages.push(BaseMessage::system(CLARIFIER_SYSTEM_PROMPT));
    for message in history.get_history() {
        if message.role == "assistant" {
            messages.push(BaseMessage::assistant(message.content.clone()));
        } else {
            messages.push(BaseMessage::user(message.content.clone()));
        }
    }
    messages.push(BaseMessage::user(query));
    messages
}

/// Parses a model answer into a [`Clarification`].
///
/// Failures are logged and reported as the reason the caller falls back to the
/// deterministic heuristic with.
fn parse_answer(text: &str) -> std::result::Result<Clarification, &'static str> {
    let Some(value) = extract_json_fence(text) else {
        tracing::debug!("requirement clarifier answer has no JSON fence");
        return Err(MODEL_NO_FENCE);
    };
    match serde_json::from_value::<ClarificationAnswer>(value) {
        Ok(answer) => Ok(Clarification {
            ready: answer.ready,
            question: answer.question.trim().to_owned(),
            summary: answer.summary.trim().to_owned(),
            provenance: GenerationProvenance::new(),
        }),
        Err(error) => {
            tracing::debug!(%error, "requirement clarifier answer does not match the contract");
            Err(MODEL_CONTRACT_MISMATCH)
        }
    }
}

/// The deterministic keyword heuristic documented in the module docs.
fn deterministic(query: &str, history: &HistoryManager) -> Clarification {
    let context = context_text(query, history);
    let lower = context.to_lowercase();

    let goal = contains_any(&lower, GOAL_KEYWORDS);
    let input = contains_any(&lower, INPUT_KEYWORDS) || lower.contains("${");
    let output = contains_any(&lower, OUTPUT_KEYWORDS);
    let tools = contains_any(&lower, TOOLS_KEYWORDS);

    let ready = goal && output;
    let question = if ready {
        String::new()
    } else if !goal {
        GOAL_QUESTION.to_owned()
    } else if !input {
        INPUT_QUESTION.to_owned()
    } else if !output {
        OUTPUT_QUESTION.to_owned()
    } else {
        TOOLS_QUESTION.to_owned()
    };

    let mut signals = Vec::with_capacity(4);
    for (name, present) in [
        ("goal", goal),
        ("input", input),
        ("output", output),
        ("tools", tools),
    ] {
        if present {
            signals.push(name);
        }
    }

    Clarification {
        ready,
        question,
        summary: signals.join(", "),
        provenance: GenerationProvenance::new(),
    }
}

/// The retained user turns, oldest first, followed by the query, all joined
/// with newlines.
fn context_text(query: &str, history: &HistoryManager) -> String {
    let mut parts: Vec<&str> = history
        .get_history()
        .iter()
        .filter(|message| message.role == "user")
        .map(|message| message.content.as_str())
        .collect();
    parts.push(query);
    parts.join("\n")
}

/// Whether `haystack` contains any of `needles`.
fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};

    /// A model answer naming a missing detail.
    const NEEDS_MODEL_FENCE: &str = r#"```json
{"ready": false, "question": "Which model?", "summary": "needs a model"}
```"#;

    /// Coerces a scripted model into the trait object the clarifier takes.
    fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
        let cloned: Arc<ScriptedModel> = Arc::clone(model);
        cloned
    }

    /// One `clarifying` provenance record.
    fn provenance_of(source: GenerationSource, detail: &str) -> GenerationProvenance {
        let mut provenance = GenerationProvenance::new();
        provenance.record(ProgressStage::Clarifying, source, detail);
        provenance
    }

    /// The three Python-visible verdict fields, for comparing paths that differ
    /// only in provenance.
    fn verdict(clarification: &Clarification) -> (bool, String, String) {
        (
            clarification.ready,
            clarification.question.clone(),
            clarification.summary.clone(),
        )
    }

    /// A history holding one request that states a goal and an output.
    fn goal_and_output_history() -> HistoryManager {
        let mut history = HistoryManager::new(20);
        history.add_user_message("Build an agent that summarizes documents and returns a summary.");
        history
    }

    #[tokio::test]
    async fn blank_queries_are_validation_errors() {
        let history = HistoryManager::new(20);
        for query in ["", "   ", "\n\t"] {
            let error = Clarifier::new()
                .clarify(query, &history)
                .await
                .expect_err("blank query is rejected");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.to_string().contains("query must not be blank"));
        }
    }

    #[tokio::test]
    async fn blank_query_with_a_model_never_calls_it() {
        let model = Arc::new(ScriptedModel::text(NEEDS_MODEL_FENCE));
        let clarifier = Clarifier::with_model(dynamic(&model));
        let history = HistoryManager::new(20);

        let error = clarifier
            .clarify("  ", &history)
            .await
            .expect_err("blank query is rejected");

        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(model.call_count(), 0);
    }

    #[tokio::test]
    async fn complete_request_is_ready() {
        let history = HistoryManager::new(20);
        let clarification = Clarifier::new()
            .clarify(
                "Create a customer service assistant that answers product questions.",
                &history,
            )
            .await
            .expect("clarify succeeds");

        assert!(clarification.ready);
        assert_eq!(clarification.question, "");
        assert_eq!(clarification.summary, "goal, input, output");
        assert!(clarification.summary.contains("goal"));
        assert!(clarification.summary.contains("output"));
    }

    #[tokio::test]
    async fn missing_output_asks_for_the_input_first() {
        let history = HistoryManager::new(20);
        let clarification = Clarifier::new()
            .clarify("Create an assistant.", &history)
            .await
            .expect("clarify succeeds");

        assert!(!clarification.ready);
        assert_eq!(clarification.question, INPUT_QUESTION);
        assert_eq!(clarification.summary, "goal");
    }

    #[tokio::test]
    async fn missing_tools_alone_is_ready() {
        let history = HistoryManager::new(20);
        let clarification = Clarifier::new()
            .clarify(
                "Build an agent that reads the input and returns a summary.",
                &history,
            )
            .await
            .expect("clarify succeeds");

        assert!(clarification.ready);
        assert_eq!(clarification.question, "");
        assert_eq!(clarification.summary, "goal, input, output");
    }

    #[tokio::test]
    async fn a_request_without_a_goal_asks_for_it() {
        let history = HistoryManager::new(20);
        let clarification = Clarifier::new()
            .clarify("Something with a database.", &history)
            .await
            .expect("clarify succeeds");

        assert!(!clarification.ready);
        assert_eq!(clarification.question, GOAL_QUESTION);
        assert_eq!(clarification.summary, "tools");
    }

    #[tokio::test]
    async fn history_user_turns_contribute_signals() {
        let history = goal_and_output_history();

        let alone = Clarifier::new()
            .clarify("please continue", &HistoryManager::new(20))
            .await
            .expect("clarify succeeds");
        assert!(!alone.ready);
        assert_eq!(alone.question, GOAL_QUESTION);
        assert_eq!(alone.summary, "");

        let with_history = Clarifier::new()
            .clarify("please continue", &history)
            .await
            .expect("clarify succeeds");
        assert!(with_history.ready);
        assert_eq!(with_history.question, "");
        assert_eq!(with_history.summary, "goal, input, output");
    }

    #[tokio::test]
    async fn model_answer_is_parsed_and_the_conversation_ordered() {
        let model = Arc::new(ScriptedModel::text(NEEDS_MODEL_FENCE));
        let clarifier = Clarifier::with_model(dynamic(&model));

        let mut history = HistoryManager::new(20);
        history.add_user_message("I need a helper.");
        history.add_assistant_message("What should it do?");

        let clarification = clarifier
            .clarify("It should answer questions.", &history)
            .await
            .expect("clarify succeeds");

        assert_eq!(
            clarification,
            Clarification {
                ready: false,
                question: "Which model?".to_string(),
                summary: "needs a model".to_string(),
                provenance: provenance_of(GenerationSource::Model, ""),
            }
        );
        assert_eq!(model.call_count(), 1);
        let call = model.last_call().expect("one call");
        assert_eq!(call.messages.len(), 4);
        assert_eq!(call.messages[0].text(), CLARIFIER_SYSTEM_PROMPT);
        assert_eq!(call.messages[1].text(), "I need a helper.");
        assert_eq!(call.messages[2].text(), "What should it do?");
        assert!(call.messages[3]
            .text()
            .contains("It should answer questions."));
        assert_eq!(call.request.model_name, "scripted-model");
    }

    #[tokio::test]
    async fn model_answer_strings_are_trimmed() {
        let model = Arc::new(ScriptedModel::text(
            "```json\n{\"ready\": true, \"question\": \"  \", \"summary\": \" goal, output \"}\n```",
        ));
        let clarifier = Clarifier::with_model(dynamic(&model));

        let clarification = clarifier
            .clarify(
                "Build an agent that returns a result.",
                &HistoryManager::new(20),
            )
            .await
            .expect("clarify succeeds");

        assert_eq!(
            clarification,
            Clarification {
                ready: true,
                question: String::new(),
                summary: "goal, output".to_string(),
                provenance: provenance_of(GenerationSource::Model, ""),
            }
        );
        assert_eq!(model.call_count(), 1);
    }

    #[tokio::test]
    async fn prose_answer_falls_back_to_the_heuristic() {
        let model = Arc::new(ScriptedModel::text("I think it is probably fine."));
        let clarifier = Clarifier::with_model(dynamic(&model));
        let query = "Create a customer service assistant that answers product questions.";
        let history = HistoryManager::new(20);

        let from_model = clarifier
            .clarify(query, &history)
            .await
            .expect("clarify succeeds");
        let from_heuristic = Clarifier::new()
            .clarify(query, &history)
            .await
            .expect("clarify succeeds");

        assert_eq!(verdict(&from_model), verdict(&from_heuristic));
        assert_eq!(
            from_model.provenance,
            provenance_of(GenerationSource::Fallback, MODEL_NO_FENCE)
        );
        assert_eq!(
            from_heuristic.provenance,
            provenance_of(GenerationSource::Deterministic, NO_MODEL_DETAIL)
        );
        assert!(from_model.provenance.has_fallback());
        assert_eq!(model.call_count(), 1);
    }

    #[tokio::test]
    async fn json_without_the_contract_falls_back() {
        for answer in [
            "```json\n[]\n```",
            "```json\n{\"question\": \"Which model?\"}\n```",
            "```json\n{\"ready\": \"yes\"}\n```",
        ] {
            let model = Arc::new(ScriptedModel::text(answer));
            let clarifier = Clarifier::with_model(dynamic(&model));
            let history = HistoryManager::new(20);

            let clarification = clarifier
                .clarify("Build an agent.", &history)
                .await
                .expect("clarify succeeds");

            let heuristic = Clarifier::new()
                .clarify("Build an agent.", &history)
                .await
                .expect("clarify succeeds");
            assert_eq!(verdict(&clarification), verdict(&heuristic));
            assert_eq!(
                clarification.provenance.stages[0].source,
                GenerationSource::Fallback
            );
            assert_eq!(model.call_count(), 1);
        }
    }

    #[tokio::test]
    async fn model_failure_falls_back_to_the_heuristic() {
        let model = Arc::new(ScriptedModel::new([ScriptedTurn::fail("boom")]));
        let clarifier = Clarifier::with_model(dynamic(&model));
        let history = HistoryManager::new(20);

        let clarification = clarifier
            .clarify("Build an agent that reads the input.", &history)
            .await
            .expect("a model failure is not an error");

        assert!(!clarification.ready);
        assert_eq!(clarification.question, OUTPUT_QUESTION);
        assert_eq!(
            clarification.provenance,
            provenance_of(GenerationSource::Fallback, MODEL_CALL_FAILED)
        );
        let heuristic = Clarifier::new()
            .clarify("Build an agent that reads the input.", &history)
            .await
            .expect("clarify succeeds");
        assert_eq!(verdict(&clarification), verdict(&heuristic));
        assert_eq!(model.call_count(), 1);
    }

    #[test]
    fn default_and_new_are_offline() {
        assert_eq!(
            format!("{:?}", Clarifier::new()),
            "Clarifier { has_model: false }"
        );
        assert_eq!(
            format!("{:?}", Clarifier::default()),
            "Clarifier { has_model: false }"
        );

        let model = Arc::new(ScriptedModel::text(NEEDS_MODEL_FENCE));
        assert_eq!(
            format!("{:?}", Clarifier::with_model(dynamic(&model))),
            "Clarifier { has_model: true }"
        );
    }

    #[test]
    fn clarification_round_trips_through_snake_case_json() {
        let clarification = Clarification {
            ready: false,
            question: "Which model?".to_string(),
            summary: "needs a model".to_string(),
            provenance: provenance_of(GenerationSource::Model, ""),
        };

        let value = serde_json::to_value(&clarification).expect("serialize");
        assert_eq!(
            value,
            serde_json::json!({
                "ready": false,
                "question": "Which model?",
                "summary": "needs a model",
                "provenance": {"stages": [
                    {"stage": "clarifying", "source": "model", "detail": ""},
                ]},
            })
        );

        let parsed: Clarification = serde_json::from_value(value).expect("deserialize");
        assert_eq!(parsed, clarification);
        assert_eq!(
            Clarification::default(),
            Clarification {
                ready: false,
                question: String::new(),
                summary: String::new(),
                provenance: GenerationProvenance::new(),
            }
        );

        // Legacy JSON without `provenance` still parses, with no recorded stage.
        let legacy: Clarification = serde_json::from_value(serde_json::json!({
            "ready": true,
            "question": "",
            "summary": "goal, output",
        }))
        .expect("legacy JSON parses");
        assert!(legacy.ready);
        assert!(legacy.provenance.is_empty());
    }
}
