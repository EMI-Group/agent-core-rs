//! Workflow selection: single-workflow fast path plus LLM intent detection.
//!
//! Python parity: `WorkflowAgent._select_workflow` in
//! `openjiuwen/core/application/workflow_agent/`.
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{BaseMessage, Model, ModelRequestConfig};
use openjiuwen_workflow::Workflow;
use std::sync::Arc;

/// The routing decision for one invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Selection {
    /// Run the workflow at this index.
    Workflow(usize),
    /// Answer with the configured default response instead of running a
    /// workflow.
    Default(String),
}

/// The intent-detection instruction; the reply is parsed for a 1-based index.
const SYSTEM_PROMPT: &str =
    "You route user requests to one workflow. Reply with JSON {\"result\": <n>}.";

/// Picks the workflow that should answer `query`.
///
/// - an empty slice is an error (`No workflows configured for single_agent`);
/// - exactly one workflow always wins (index `0`) and the model is never called;
/// - with several workflows and a model, a numbered list of each workflow's
///   description (falling back to its name when the description is empty) is
///   sent to the model, and the first run of digits in the reply is read as a
///   1-based index;
/// - any detection failure (no model, provider error, unparsable or
///   out-of-range reply) falls back to `default_text` when set, else index `0`.
///
/// # Errors
///
/// Returns [`Error::workflow`] when `workflows` is empty. Model failures are
/// absorbed by the fallback and do not surface.
pub(crate) async fn select(
    workflows: &[Arc<Workflow>],
    query: &str,
    model: Option<&Arc<dyn Model>>,
    request: &ModelRequestConfig,
    default_text: Option<&str>,
) -> Result<Selection> {
    if workflows.is_empty() {
        return Err(Error::workflow("No workflows configured for single_agent"));
    }
    if workflows.len() == 1 {
        return Ok(Selection::Workflow(0));
    }
    let Some(model) = model else {
        return Ok(fallback(default_text));
    };
    let messages = [
        BaseMessage::system(SYSTEM_PROMPT),
        BaseMessage::user(user_prompt(workflows, query)),
    ];
    let reply = match model.invoke(&messages, request).await {
        Ok(reply) => reply,
        Err(error) => {
            tracing::debug!(%error, "workflow selection model call failed; falling back");
            return Ok(fallback(default_text));
        }
    };
    let text = reply
        .message()
        .map(|message| message.text().into_owned())
        .unwrap_or_default();
    Ok(parse_selection(&text, workflows.len())
        .map_or_else(|| fallback(default_text), Selection::Workflow))
}

/// The numbered `description or name` list plus the query.
fn user_prompt(workflows: &[Arc<Workflow>], query: &str) -> String {
    let listing = workflows
        .iter()
        .enumerate()
        .map(|(position, workflow)| {
            let card = workflow.card();
            let label = if card.description.is_empty() {
                &card.name
            } else {
                &card.description
            };
            format!("{}. {label}", position + 1)
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("{listing}\n\nQuery: {query}")
}

/// Reads the first run of digits in `text` as a 1-based index within `count`.
fn parse_selection(text: &str, count: usize) -> Option<usize> {
    let start = text.find(|character: char| character.is_ascii_digit())?;
    let digits = text[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    let number = digits.parse::<usize>().ok()?;
    (1..=count).contains(&number).then(|| number - 1)
}

/// The failure path: the configured default response, else the first workflow.
fn fallback(default_text: Option<&str>) -> Selection {
    default_text.map_or(Selection::Workflow(0), |text| {
        Selection::Default(text.to_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_test_support::ScriptedModel;
    use openjiuwen_workflow::card::WorkflowCard;

    fn workflow(id: &str) -> Arc<Workflow> {
        Arc::new(Workflow::new(WorkflowCard::new(id)))
    }

    fn described(id: &str, description: &str) -> Arc<Workflow> {
        let mut card = WorkflowCard::new(id);
        card.description = description.to_owned();
        Arc::new(Workflow::new(card))
    }

    fn shared(model: &ScriptedModel) -> Arc<dyn Model> {
        Arc::new(model.clone())
    }

    fn request() -> ModelRequestConfig {
        ModelRequestConfig::new("test")
    }

    #[tokio::test]
    async fn empty_slice_is_a_workflow_error() {
        let error = select(&[], "hi", None, &request(), None).await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Workflow);
        assert!(error.to_string().contains("No workflows configured"));
    }

    #[tokio::test]
    async fn single_workflow_never_calls_the_model() {
        let scripted = ScriptedModel::text("2");
        let model = shared(&scripted);
        let workflows = vec![workflow("only")];
        assert_eq!(
            select(&workflows, "hi", Some(&model), &request(), None)
                .await
                .unwrap(),
            Selection::Workflow(0)
        );
        assert_eq!(scripted.call_count(), 0);
    }

    #[tokio::test]
    async fn json_reply_selects_the_index() {
        let scripted = ScriptedModel::text(r#"{"result": 2}"#);
        let model = shared(&scripted);
        let workflows = vec![workflow("a"), workflow("b"), workflow("c")];
        assert_eq!(
            select(&workflows, "hi", Some(&model), &request(), None)
                .await
                .unwrap(),
            Selection::Workflow(1)
        );
        assert_eq!(scripted.call_count(), 1);
    }

    #[tokio::test]
    async fn bare_and_quoted_numbers_are_parsed() {
        let bare = ScriptedModel::text("2");
        let bare_model = shared(&bare);
        let workflows = vec![workflow("a"), workflow("b"), workflow("c")];
        assert_eq!(
            select(&workflows, "hi", Some(&bare_model), &request(), None)
                .await
                .unwrap(),
            Selection::Workflow(1)
        );

        let quoted = ScriptedModel::text(r#""3""#);
        let quoted_model = shared(&quoted);
        assert_eq!(
            select(&workflows, "hi", Some(&quoted_model), &request(), None)
                .await
                .unwrap(),
            Selection::Workflow(2)
        );
    }

    #[tokio::test]
    async fn detection_prompt_lists_descriptions_or_names() {
        let scripted = ScriptedModel::text("1");
        let model = shared(&scripted);
        let workflows = vec![described("a", "Book a flight"), workflow("b")];
        select(&workflows, "book me", Some(&model), &request(), None)
            .await
            .unwrap();
        let call = scripted.last_call().expect("one model call");
        let prompt = call.last_user_text().expect("a user message");
        assert!(prompt.contains("1. Book a flight"));
        assert!(prompt.contains("2. b"));
        assert!(prompt.ends_with("Query: book me"));
    }

    #[tokio::test]
    async fn garbage_reply_falls_back_to_the_first_workflow() {
        let scripted = ScriptedModel::text("I cannot help with that");
        let model = shared(&scripted);
        let workflows = vec![workflow("a"), workflow("b")];
        assert_eq!(
            select(&workflows, "hi", Some(&model), &request(), None)
                .await
                .unwrap(),
            Selection::Workflow(0)
        );
    }

    #[tokio::test]
    async fn garbage_reply_falls_back_to_the_default_text() {
        let scripted = ScriptedModel::text("no idea");
        let model = shared(&scripted);
        let workflows = vec![workflow("a"), workflow("b")];
        assert_eq!(
            select(&workflows, "hi", Some(&model), &request(), Some("fallback"))
                .await
                .unwrap(),
            Selection::Default("fallback".to_owned())
        );
    }

    #[tokio::test]
    async fn out_of_range_reply_falls_back() {
        for reply in ["0", r#"{"result": 9}"#] {
            let scripted = ScriptedModel::text(reply);
            let model = shared(&scripted);
            let workflows = vec![workflow("a"), workflow("b")];
            assert_eq!(
                select(&workflows, "hi", Some(&model), &request(), Some("fallback"))
                    .await
                    .unwrap(),
                Selection::Default("fallback".to_owned()),
                "reply {reply:?}"
            );
        }
    }

    #[tokio::test]
    async fn model_error_falls_back() {
        let scripted = ScriptedModel::new([openjiuwen_test_support::ScriptedTurn::fail("boom")]);
        let model = shared(&scripted);
        let workflows = vec![workflow("a"), workflow("b")];
        assert_eq!(
            select(&workflows, "hi", Some(&model), &request(), Some("fallback"))
                .await
                .unwrap(),
            Selection::Default("fallback".to_owned())
        );
        assert_eq!(scripted.call_count(), 1);
    }

    #[tokio::test]
    async fn no_model_falls_back() {
        let workflows = vec![workflow("a"), workflow("b")];
        assert_eq!(
            select(&workflows, "hi", None, &request(), Some("fallback"))
                .await
                .unwrap(),
            Selection::Default("fallback".to_owned())
        );
        assert_eq!(
            select(&workflows, "hi", None, &request(), None)
                .await
                .unwrap(),
            Selection::Workflow(0)
        );
    }
}
