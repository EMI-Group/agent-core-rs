//! Port of `openjiuwen/harness/tools/ask_user.py`: the `ask_user` tool and its user-asker seam.
//!
//! Python's `AskUserTool` simply returns `{}`; the questions reach the human through the
//! `AskUserRail` interrupt. Here the tool owns the payload contract (question count and option
//! validation) and delegates the actual interaction to an injected [`UserAsker`], so the rail
//! can be wired in `crate::rails` without the tool depending on it.

// The module stays private until `ToolRegistry::build` wires the tool set; keeping the
// intended public surface (`pub`) documents the eventual export and avoids a churn diff.
#![allow(unreachable_pub)]

use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};

use super::descriptions;
use super::support::{parse_args, schema_for, tool_card, ToolPayload};

/// The largest number of questions one `ask_user` call accepts (Python "1-4 questions").
const MAX_QUESTIONS: usize = 4;

/// The failure reported when no user-interaction handler is wired into the tool.
const NO_ASKER_MESSAGE: &str =
    "ask_user is not available: no user interaction handler is configured";

/// One selectable option of an [`AskQuestion`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AskOption {
    /// Short label the user sees and selects.
    pub label: String,
    /// One-line explanation of what the option means.
    pub description: String,
    /// Optional rich preview, used only for single-select questions with visual comparison.
    #[serde(default)]
    pub preview: Option<String>,
}

/// One question raised with the user through [`UserAsker`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AskQuestion {
    /// Short header naming the question; also used in validation messages.
    pub header: String,
    /// The question text shown to the user.
    pub question: String,
    /// The selectable options; at least one is required.
    pub options: Vec<AskOption>,
    /// Whether the user may select several options at once.
    #[serde(default)]
    pub multi_select: bool,
}

/// The seam between the `ask_user` tool and the user-interaction rail.
///
/// Python parity: the questions flow through the `AskUserRail` interrupt; this trait is the
/// point where the harness plugs a real prompt (REPL, HITL channel, test double) in.
#[async_trait]
pub trait UserAsker: Send + Sync {
    /// Raises the questions with the user and returns the answer payload.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the interaction fails or is
    /// cancelled; the tool forwards the message as a `success=false` payload.
    async fn ask(&self, questions: Vec<AskQuestion>) -> Result<Value>;
}

/// Arguments of the `ask_user` tool.
///
/// Python parity: the `questions` parameter of the Python `ask_user` schema. The card renders
/// the bounds and the nested question/option shape inline (see [`install_ask_questions`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AskUserArgs {
    /// The questions to raise with the user; between one and four per call.
    #[schemars(length(min = 1, max = 4))]
    pub questions: Vec<AskQuestion>,
}

/// Asks the human user one or more questions (`ask_user`).
pub struct AskUserTool {
    /// The interaction handler, when one is wired in.
    asker: Option<Arc<dyn UserAsker>>,
    /// The model-facing card built from the description language.
    card: ToolCard,
}

impl AskUserTool {
    /// Builds the tool over `asker` with the description language `language`.
    ///
    /// A `None` asker is legal: [`Tool::invoke`] then reports the Python
    /// "no user interaction handler" failure instead of hanging.
    #[must_use]
    pub fn new(asker: Option<Arc<dyn UserAsker>>, language: &str) -> Self {
        Self {
            asker,
            card: Self::card_for(language),
        }
    }

    /// Builds the model-facing card in `language`.
    ///
    /// Unknown languages fall back to the Python default (`cn`) inside
    /// [`descriptions::description`].
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        let mut input_params = schema_for::<AskUserArgs>();
        install_ask_questions(&mut input_params);
        tool_card(
            "ask_user",
            descriptions::description("ask_user", language),
            input_params,
            true,
        )
    }

    /// Validates `inputs` and raises the questions through the wired asker.
    async fn ask(&self, inputs: &Value) -> ToolPayload {
        // Python reads `inputs.get("questions")` directly and reports an absent or empty list
        // with the plain validation text rather than a schema error, so check it before
        // parsing. Non-object input is left to `parse_args` so the protocol failure wins.
        if inputs.is_object()
            && inputs
                .get("questions")
                .and_then(Value::as_array)
                .is_none_or(Vec::is_empty)
        {
            return ToolPayload::fail("questions is required");
        }

        let args: AskUserArgs = match parse_args(inputs) {
            Ok(args) => args,
            Err(payload) => return payload,
        };

        if args.questions.len() > MAX_QUESTIONS {
            return ToolPayload::fail(format!("questions supports at most {MAX_QUESTIONS} items"));
        }

        if let Some(question) = args
            .questions
            .iter()
            .find(|question| question.options.is_empty())
        {
            return ToolPayload::fail(format!(
                "question '{}' must provide at least one option",
                question.header
            ));
        }

        let Some(asker) = self.asker.as_ref() else {
            return ToolPayload::fail(NO_ASKER_MESSAGE);
        };

        // Python returns `{}` on success, so the asker's answer payload is deliberately
        // dropped: the rail owns how answers are surfaced to the model. On failure only the
        // error message is forwarded (no `tool: ` kind prefix), matching the Python text.
        match asker.ask(args.questions).await {
            Ok(_) => ToolPayload::ok_empty(),
            Err(error) => ToolPayload::fail(error.message().into_owned()),
        }
    }
}

#[async_trait]
impl Tool for AskUserTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.ask(&inputs).await.to_json())
    }
}

/// Replaces the `questions` property of an [`AskUserArgs`] schema with an inline definition.
///
/// [`AskQuestion`] and [`AskOption`] are named types, so schemars describes them as
/// `$ref: "#/$defs/..."` entries; [`schema_for`] strips `$defs`, which would leave dangling
/// references in the card. The card carries the Python question/option shape inline instead.
fn install_ask_questions(schema: &mut Value) {
    let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut) else {
        return;
    };
    properties.insert(
        "questions".to_owned(),
        json!({
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_QUESTIONS,
            "items": {
                "type": "object",
                "properties": {
                    "header": {"type": "string"},
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "description": {"type": "string"},
                                "preview": {"type": "string"}
                            },
                            "required": ["label", "description"]
                        }
                    },
                    "multi_select": {"type": "boolean", "default": false}
                },
                "required": ["header", "question", "options"]
            }
        }),
    );
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Mutex;

    use openjiuwen_common::error::Error;

    use super::*;

    /// A question with `header` and a single usable option.
    fn question(header: &str) -> Value {
        json!({
            "header": header,
            "question": "How should this proceed?",
            "options": [{"label": "A", "description": "first"}],
        })
    }

    /// Runs `invoke` with a fresh, uncancelled context and unwraps the payload.
    async fn invoke(tool: &AskUserTool, inputs: Value) -> Value {
        Tool::invoke(tool, inputs, &ToolContext::new())
            .await
            .unwrap()
    }

    /// A [`UserAsker`] that records the questions and answers with a fixed payload.
    struct RecordingAsker {
        /// The questions seen by the asker, in call order.
        seen: Arc<Mutex<Vec<AskQuestion>>>,
    }

    #[async_trait]
    impl UserAsker for RecordingAsker {
        async fn ask(&self, questions: Vec<AskQuestion>) -> Result<Value> {
            self.seen.lock().unwrap().extend(questions);
            Ok(json!({"answers": []}))
        }
    }

    /// A [`UserAsker`] whose interaction always fails.
    struct FailingAsker;

    #[async_trait]
    impl UserAsker for FailingAsker {
        async fn ask(&self, _questions: Vec<AskQuestion>) -> Result<Value> {
            Err(Error::tool("boom"))
        }
    }

    #[test]
    fn card_carries_the_python_schema_without_dangling_refs() {
        let card = AskUserTool::card_for("en");

        assert_eq!(card.name, "ask_user");
        assert!(card.parallel_safe);
        assert_eq!(
            card.description,
            descriptions::description("ask_user", "en")
        );

        let schema = &card.input_params;
        let rendered = schema.to_string();
        assert!(
            !rendered.contains("$ref"),
            "dangling reference in {rendered}"
        );

        assert_eq!(schema["type"], json!("object"));
        assert_eq!(schema["required"], json!(["questions"]));
        let properties = schema["properties"].as_object().unwrap();
        assert_eq!(
            properties.keys().collect::<Vec<_>>(),
            vec!["questions"],
            "unexpected property set"
        );

        let questions = &schema["properties"]["questions"];
        assert_eq!(questions["type"], json!("array"));
        assert_eq!(questions["minItems"], json!(1));
        assert_eq!(questions["maxItems"], json!(4));
        assert_eq!(
            questions["items"]["required"],
            json!(["header", "question", "options"])
        );
        assert_eq!(
            questions["items"]["properties"]["multi_select"]["type"],
            json!("boolean")
        );
        assert_eq!(
            questions["items"]["properties"]["options"]["items"]["required"],
            json!(["label", "description"])
        );
    }

    #[test]
    fn card_falls_back_to_the_chinese_description_for_unknown_languages() {
        let cn = AskUserTool::card_for("cn");
        assert_eq!(cn.description, descriptions::description("ask_user", "cn"));

        let unknown = AskUserTool::card_for("xx");
        assert_eq!(
            unknown.description,
            descriptions::description("ask_user", "cn")
        );
        assert_ne!(cn.description, AskUserTool::card_for("en").description);
    }

    #[test]
    fn schema_for_args_carries_the_question_bounds() {
        let schema = schema_for::<AskUserArgs>();
        assert_eq!(schema["required"], json!(["questions"]));
        assert_eq!(schema["properties"]["questions"]["minItems"], json!(1));
        assert_eq!(schema["properties"]["questions"]["maxItems"], json!(4));
    }

    #[tokio::test]
    async fn missing_questions_is_the_plain_validation_message() {
        let tool = AskUserTool::new(None, "en");

        for inputs in [json!({}), json!({"questions": []})] {
            let out = invoke(&tool, inputs).await;
            assert_eq!(out["success"], json!(false));
            assert_eq!(out["error"], json!("questions is required"));
        }
    }

    #[tokio::test]
    async fn non_object_arguments_are_a_protocol_failure() {
        let tool = AskUserTool::new(None, "en");

        let out = invoke(&tool, json!([])).await;
        assert_eq!(out["success"], json!(false));
        assert_eq!(out["error"], json!("tool arguments must be an object"));
    }

    #[tokio::test]
    async fn more_than_four_questions_is_rejected() {
        let tool = AskUserTool::new(None, "en");
        let questions = (0..5).map(|_| question("q")).collect::<Vec<_>>();

        let out = invoke(&tool, json!({"questions": questions})).await;
        assert_eq!(out["success"], json!(false));
        assert_eq!(out["error"], json!("questions supports at most 4 items"));
    }

    #[tokio::test]
    async fn a_question_without_options_is_rejected() {
        let tool = AskUserTool::new(None, "en");

        let out = invoke(
            &tool,
            json!({
                "questions": [{
                    "header": "Pick one",
                    "question": "Which?",
                    "options": [],
                }],
            }),
        )
        .await;

        assert_eq!(out["success"], json!(false));
        assert_eq!(
            out["error"],
            json!("question 'Pick one' must provide at least one option")
        );
    }

    #[tokio::test]
    async fn a_missing_asker_is_a_failure_payload() {
        let tool = AskUserTool::new(None, "en");

        let out = invoke(&tool, json!({"questions": [question("q")]})).await;

        assert_eq!(out["success"], json!(false));
        assert_eq!(
            out["error"],
            json!("ask_user is not available: no user interaction handler is configured")
        );
        assert_eq!(out["data"], Value::Null);
    }

    #[tokio::test]
    async fn the_asker_receives_the_questions_and_the_payload_is_empty() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let tool = AskUserTool::new(
            Some(Arc::new(RecordingAsker {
                seen: Arc::clone(&seen),
            })),
            "en",
        );
        let sent = vec![question("first"), question("second")];

        let out = invoke(&tool, json!({"questions": sent})).await;

        assert_eq!(
            *seen.lock().unwrap(),
            vec![
                AskQuestion {
                    header: "first".to_owned(),
                    question: "How should this proceed?".to_owned(),
                    options: vec![AskOption {
                        label: "A".to_owned(),
                        description: "first".to_owned(),
                        preview: None,
                    }],
                    multi_select: false,
                },
                AskQuestion {
                    header: "second".to_owned(),
                    question: "How should this proceed?".to_owned(),
                    options: vec![AskOption {
                        label: "A".to_owned(),
                        description: "first".to_owned(),
                        preview: None,
                    }],
                    multi_select: false,
                },
            ]
        );

        assert_eq!(out["success"], json!(true));
        assert_eq!(out["data"], json!({}), "Python returns an empty object");
        assert_eq!(out["error"], Value::Null);
        for key in [
            "success",
            "data",
            "error",
            "extracted_content",
            "include_extracted_content_only_once",
            "long_term_memory",
        ] {
            assert!(out.get(key).is_some(), "missing payload key {key}");
        }
    }

    #[tokio::test]
    async fn an_asker_error_becomes_a_failure_payload() {
        let tool = AskUserTool::new(Some(Arc::new(FailingAsker)), "en");

        let out = invoke(&tool, json!({"questions": [question("q")]})).await;

        assert_eq!(out["success"], json!(false));
        assert_eq!(out["error"], json!("boom"));
    }

    #[tokio::test]
    async fn a_cancelled_context_returns_the_cancelled_error() {
        let tool = AskUserTool::new(None, "en");
        let ctx = ToolContext::new();
        ctx.cancellation.cancel();

        let error = Tool::invoke(&tool, json!({"questions": [question("q")]}), &ctx)
            .await
            .unwrap_err();

        assert!(matches!(error, Error::Cancelled));
    }
}
