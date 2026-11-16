//! The staged pipeline behind [`AgentBuilder::build_with_pipeline`].
//!
//! The pipeline is one event stream ([`super::stream::pipeline_stream`]) and two
//! drivers over it: this module's [`run`] collapses the stream to its
//! [`PipelineEvent::Final`] result for the non-streaming entry point, while
//! [`AgentBuilder::build_with_pipeline_stream`] hands the same stream to the
//! caller.
//!
//! The stages, in order, are:
//!
//! 1. intention detection — [`IntentionDetector`], model-assisted only when the
//!    session already holds turns, otherwise deterministic;
//! 2. refinement — when the session holds a workflow artifact and the detector
//!    reports [`BuildIntent::Refine`], one `DlGenerator::refine` round trip;
//! 3. clarification — [`Clarifier::clarify_rounds`] over a working copy of the
//!    retained history, stopping the build with the question when not ready;
//! 4. for a workflow: the three-pass designer, the DL generator, and the bounded
//!    repair loop that reflects, refines, and reflects again — starting from the
//!    model's document even when it did not validate, so the loop has something
//!    to repair;
//! 5. for an LLM agent: one generation round trip plus the deterministic
//!    document repair.
//!
//! Every stage appends to one [`GenerationProvenance`], the repair loop owns the
//! returned [`RepairReport`], and a model failure always degrades to the
//! documented deterministic path instead of erroring.
//!
//! [`AgentBuilder::build_with_pipeline`]: super::AgentBuilder::build_with_pipeline
//! [`AgentBuilder::build_with_pipeline_stream`]: super::AgentBuilder::build_with_pipeline_stream

use std::sync::Arc;

use futures::StreamExt;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};

use super::super::clarifier::Clarification;
use super::super::designer::WorkflowDesign;
use super::super::dl_generator::{DlGenerator, GeneratedDl};
use super::super::dl_reflector::{ReflectionIssue, ReflectionIssueKind};
use super::super::dsl::LlmAgentDsl;
use super::super::history::HistoryManager;
use super::super::outcome::{BuildResult, GenerationProvenance, GenerationSource};
use super::super::workflow_dsl::WorkflowDsl;
use super::stream::{pipeline_stream, PipelineEvent};
use super::{
    conversation, deterministic_agent_dsl, repair_llm_dsl, AgentType,
    DEFAULT_CLARIFICATION_QUESTION,
};
use crate::util::extract_json_fence;

/// Runs the staged pipeline and returns its final result.
///
/// The stream is the single implementation of the pipeline, so the non-streaming
/// result and the stream's [`PipelineEvent::Final`] are always the same object.
///
/// # Errors
///
/// Returns the error the stream yields — the blank-query [`Error::Validation`],
/// or a validation error from a stage that cannot represent its input. A model
/// failure never propagates.
pub(super) async fn run(
    model: &Arc<dyn Model>,
    query: &str,
    session_id: &str,
    agent_type: AgentType,
    prior: &HistoryManager,
    previous: Option<&WorkflowDsl>,
    timestamp_ms: i64,
) -> Result<BuildResult> {
    let mut events = pipeline_stream(
        Arc::clone(model),
        query.to_string(),
        session_id.to_string(),
        agent_type,
        prior.clone(),
        previous.cloned(),
        timestamp_ms,
    );
    while let Some(event) = events.next().await {
        if let PipelineEvent::Final(result) = event? {
            return Ok(result);
        }
    }
    Err(Error::llm("pipeline stream ended without a final result"))
}

/// The text a not-ready clarification is answered with.
///
/// The clarifier's question wins, then its detected-signal summary, then the
/// generic [`DEFAULT_CLARIFICATION_QUESTION`].
pub(super) fn clarification_text(clarification: Clarification) -> String {
    if !clarification.question.is_empty() {
        clarification.question
    } else if !clarification.summary.is_empty() {
        clarification.summary
    } else {
        DEFAULT_CLARIFICATION_QUESTION.to_string()
    }
}

/// Asks the model for one LLM-agent document and repairs it deterministically.
///
/// A failed call, a choice-less answer, or an unparseable answer is logged and
/// replaced by [`deterministic_agent_dsl`]; the returned provenance says whether
/// the document came from the model or from that fallback, so this entry point
/// never reports a model problem as an error.
pub(super) async fn llm_agent_pipeline(
    model: &Arc<dyn Model>,
    query: &str,
    history: &HistoryManager,
    timestamp_ms: i64,
) -> Result<(LlmAgentDsl, GenerationSource, String)> {
    let messages = conversation(history, AgentType::LlmAgent);
    let request = ModelRequestConfig::new(model.model_name());
    let response = match model.invoke(&messages, &request).await {
        Ok(response) => response,
        Err(error) => {
            tracing::debug!(
                %error,
                "LLM-agent model call failed; using the deterministic document"
            );
            return Ok((
                deterministic_agent_dsl(query, timestamp_ms),
                GenerationSource::Fallback,
                format!("model call failed: {error}"),
            ));
        }
    };
    let Some(choice) = response.choices.into_iter().next() else {
        tracing::debug!("LLM-agent model returned no choices; using the deterministic document");
        return Ok((
            deterministic_agent_dsl(query, timestamp_ms),
            GenerationSource::Fallback,
            "model returned no choices".to_string(),
        ));
    };
    let text = choice.message.text().into_owned();
    let parsed = extract_json_fence(&text)
        .ok_or_else(|| Error::validation("model answer contained no JSON code fence"))
        .and_then(LlmAgentDsl::from_value);
    match parsed {
        Ok(dsl) => Ok((
            repair_llm_dsl(dsl, query, timestamp_ms),
            GenerationSource::Model,
            String::new(),
        )),
        Err(error) => {
            tracing::debug!(
                %error,
                "LLM-agent answer is not a DSL document; using the deterministic document"
            );
            Ok((
                deterministic_agent_dsl(query, timestamp_ms),
                GenerationSource::Fallback,
                format!("answer was not an LLM-agent document: {error}"),
            ))
        }
    }
}

/// Generates a workflow candidate for the repair loop.
///
/// This is the generator's own candidate form
/// ([`DlGenerator::generate_candidate`]): it hands back the model's document even
/// when it fails [`WorkflowDsl::validate`], so the loop can repair it instead of
/// discarding it, together with the deterministic baseline and the provenance
/// record that says where the answer came from.
///
/// # Errors
///
/// Returns the generator's [`Error::Validation`] for a design that cannot be
/// converted at all.
pub(super) async fn generate_workflow(
    generator: &DlGenerator,
    design: &WorkflowDesign,
) -> Result<GeneratedDl> {
    generator.generate_candidate(design).await
}

/// The source and detail of the newest provenance record.
pub(super) fn last_provenance(provenance: &GenerationProvenance) -> (GenerationSource, String) {
    provenance
        .stages
        .last()
        .map_or((GenerationSource::Deterministic, String::new()), |record| {
            (record.source, record.detail.clone())
        })
}

/// The wire name of an issue kind, e.g. `"cycle"`.
fn issue_kind_name(kind: ReflectionIssueKind) -> String {
    serde_json::to_value(kind)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_else(|| "issue".to_string())
}

/// One issue rendered as `kind: message [node]`.
pub(super) fn issue_note(issue: &ReflectionIssue) -> String {
    let kind = issue_kind_name(issue.kind);
    match issue.node_id.as_deref() {
        Some(node) => format!("{kind}: {} [{node}]", issue.message),
        None => format!("{kind}: {}", issue.message),
    }
}

/// Renders the problems of one repair attempt as the feedback lines.
///
/// The generation stage's validation notes come first — they have not reached the
/// model yet — then one line per issue the deterministic reflection just repaired.
fn problem_lines(notes: &[String], issues: &[ReflectionIssue]) -> String {
    let mut lines: Vec<String> = notes.to_vec();
    if !issues.is_empty() {
        lines.push(
            issues
                .iter()
                .map(issue_note)
                .collect::<Vec<_>>()
                .join(";\n"),
        );
    }
    lines.join(";\n")
}

/// Wraps rendered problem lines in the repair request the loop sends.
///
/// Python's loop description names the detected problem and demands a
/// directed-acyclic-graph conforming rewrite; this is the same request with the
/// detected problems in place of the cycle detail.
fn repair_request(lines: &str) -> String {
    format!(
        "The workflow design may contain an invalid or cyclic structure:\n{lines}\n\
         Follow the directed-acyclic-graph (DAG) rules strictly, restructure the design if \
         needed, and make sure the workflow contains no closed loop or cycle."
    )
}

/// The refinement feedback of one repair attempt.
///
/// `notes` are the not-yet-repaired problems of the attempt — the generation
/// stage's validation notes, which a repair route has still to see — and `issues`
/// are the ones the deterministic reflection just repaired. Both reach the model,
/// so a document that never validated is refined with the reason it did not.
pub(super) fn repair_feedback(issues: &[ReflectionIssue], notes: &[String]) -> String {
    repair_request(&problem_lines(notes, issues))
}
