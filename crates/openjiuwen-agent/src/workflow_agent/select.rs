//! Workflow selection: single-workflow fast path plus LLM intent detection.
//!
//! Python parity: `WorkflowAgent._select_workflow` in
//! `openjiuwen/core/application/workflow_agent/`.
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{Model, ModelRequestConfig};
use openjiuwen_workflow::Workflow;
use std::sync::Arc;

/// Picks the index of the workflow that should answer `query`.
///
/// Target behaviour:
///
/// - an empty slice is an error (`no workflows configured`);
/// - exactly one workflow always wins (index `0`);
/// - with several workflows and a model, a numbered intent-detection prompt is
///   built over each workflow's `description`, falling back to its `name` when
///   the description is empty, and the reply is parsed from `{"result": N}` or
///   a bare integer;
/// - any detection failure (no model, provider error, unparsable or
///   out-of-range reply) falls back to index `0`.
///
/// # Errors
///
/// Returns [`Error::workflow`] when `workflows` is empty. Model failures are
/// absorbed by the fallback and do not surface.
#[allow(dead_code)] // wired into WorkflowAgent::invoke by the follow-up implementation
#[allow(clippy::unused_async)] // the signature is mandated; awaits land with intent detection
pub(crate) async fn select_workflow(
    workflows: &[Arc<Workflow>],
    _query: &str,
    _model: Option<&Arc<dyn Model>>,
    _request: &ModelRequestConfig,
) -> Result<usize> {
    if workflows.is_empty() {
        return Err(Error::workflow("no workflows configured"));
    }
    // TODO: when `workflows.len() > 1` and a model is configured, build the
    // numbered intent-detection prompt (description or name per workflow), call
    // the model with `_request`, parse the selected index, and fall back to `0`
    // on any failure.
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_workflow::card::WorkflowCard;

    fn workflow(id: &str) -> Arc<Workflow> {
        Arc::new(Workflow::new(WorkflowCard::new(id)))
    }

    #[tokio::test]
    async fn empty_slice_is_a_workflow_error() {
        let request = ModelRequestConfig::new("test");
        let error = select_workflow(&[], "hi", None, &request)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Workflow);
        assert!(error.to_string().contains("no workflows configured"));
    }

    #[tokio::test]
    async fn first_workflow_is_selected_without_a_model() {
        let workflows = vec![workflow("a"), workflow("b")];
        let request = ModelRequestConfig::new("test");
        assert_eq!(
            select_workflow(&workflows, "hi", None, &request)
                .await
                .unwrap(),
            0
        );
    }
}
