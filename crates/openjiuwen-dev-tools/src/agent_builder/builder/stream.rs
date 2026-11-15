//! The streaming form of the staged build pipeline.
//!
//! [`pipeline_stream`] is the single implementation of the pipeline: it owns the
//! model, the request, the session snapshot, and the newest workflow artifact, so
//! it is `Send + 'static` and needs no borrow of the builder.
//! Every stage it runs is announced with [`PipelineEvent::StageStarted`] and
//! [`PipelineEvent::StageCompleted`], every clarification round is reported as
//! [`PipelineEvent::Clarified`], every repair attempt as [`PipelineEvent::Repaired`],
//! and a successful run ends with exactly one [`PipelineEvent::Final`].
//!
//! [`AgentBuilder::build_with_pipeline`] drives the same stream and returns the
//! `Final` result, so both entry points produce identical builds.
//! The stream never mutates a builder session; a caller persists the outcome with
//! [`AgentBuilder::record_result`].
//!
//! [`AgentBuilder::build_with_pipeline`]: super::AgentBuilder::build_with_pipeline
//! [`AgentBuilder::record_result`]: super::AgentBuilder::record_result

use std::sync::Arc;

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use serde::{Deserialize, Serialize};

use super::super::clarifier::{
    ClarificationRound, Clarifier, ClarifyConfig, DEFAULT_MAX_CLARIFY_ROUNDS,
};
use super::super::designer::{DesignEvent, DesignPassKind, WorkflowDesign, WorkflowDesigner};
use super::super::dl_generator::DlGenerator;
use super::super::dl_reflector::{DlReflector, DEFAULT_MAX_RETRIES};
use super::super::history::HistoryManager;
use super::super::intention::{BuildIntent, IntentionDetector};
use super::super::mermaid::render_mermaid;
use super::super::outcome::{
    map_state_to_status, BuildOutput, BuildResult, BuildState, GenerationProvenance,
    GenerationSource, RepairReport,
};
use super::super::progress::ProgressStage;
use super::super::workflow_dsl::WorkflowDsl;
use super::pipeline::{
    clarification_text, generate_workflow, joined_issue_notes, last_provenance, llm_agent_pipeline,
    repair_feedback,
};
use super::AgentType;

/// One event of the staged build pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
// `Final` carries the whole result, which is the payload callers asked for.
#[allow(clippy::large_enum_variant)]
pub enum PipelineEvent {
    /// A stage is about to run.
    StageStarted {
        /// The stage that starts.
        stage: ProgressStage,
    },
    /// A stage finished, together with where its output came from.
    StageCompleted {
        /// The stage that finished.
        stage: ProgressStage,
        /// Where the stage's output came from.
        source: GenerationSource,
        /// A one-line note about the stage's result.
        detail: String,
    },
    /// One clarification round settled.
    Clarified(ClarificationRound),
    /// One repair attempt settled; the payload is the report so far.
    Repaired(RepairReport),
    /// The pipeline finished; the payload is the build result.
    Final(BuildResult),
}

/// Streams the staged pipeline over owned data.
///
/// A blank query yields one [`Error::Validation`] and no
/// [`PipelineEvent::Final`]; every other failure is absorbed by the stage that
/// produced it, exactly as in the non-streaming entry point.
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
pub(super) fn pipeline_stream(
    model: Arc<dyn Model>,
    query: String,
    session_id: String,
    agent_type: AgentType,
    prior: HistoryManager,
    previous: Option<WorkflowDsl>,
    timestamp_ms: i64,
) -> BoxStream<'static, Result<PipelineEvent>> {
    Box::pin(async_stream::try_stream! {
        if query.trim().is_empty() {
            Err(Error::validation("query must not be blank"))?;
        }
        let mut provenance = GenerationProvenance::new();
        let mut repair = RepairReport::default();

        // 1. Intention: deterministic on a first request, model-assisted later.
        yield PipelineEvent::StageStarted {
            stage: ProgressStage::DetectingIntention,
        };
        let detector = if prior.is_empty() {
            IntentionDetector::new()
        } else {
            IntentionDetector::with_model(Arc::clone(&model))
        };
        let verdict = match previous.as_ref() {
            Some(workflow) if agent_type == AgentType::Workflow => {
                detector
                    .detect_refine(&query, &prior, &render_mermaid(workflow))
                    .await?
            }
            _ => detector.detect_initial(&query, &prior).await?,
        };
        provenance.record(
            ProgressStage::DetectingIntention,
            verdict.source,
            verdict.detail.clone(),
        );
        yield PipelineEvent::StageCompleted {
            stage: ProgressStage::DetectingIntention,
            source: verdict.source,
            detail: verdict.detail.clone(),
        };

        // 2. Refinement of the workflow the session already holds.
        if verdict.intent == BuildIntent::Refine {
            if let Some(workflow) = previous.as_ref() {
                yield PipelineEvent::StageStarted {
                    stage: ProgressStage::RefiningDl,
                };
                let generator = DlGenerator::with_model(Arc::clone(&model));
                let refinement = generator
                    .refine(&query, workflow, &render_mermaid(workflow))
                    .await?;
                let (source, detail) = last_provenance(&refinement.provenance);
                provenance.extend_from(refinement.provenance);
                yield PipelineEvent::StageCompleted {
                    stage: ProgressStage::RefiningDl,
                    source,
                    detail,
                };
                provenance.warn_if_degraded();
                yield PipelineEvent::Final(build_result(
                    &session_id,
                    agent_type,
                    BuildOutput::Workflow(refinement.workflow),
                    BuildState::Completed,
                    &provenance,
                    &repair,
                ));
                return;
            }
        }

        // 3. Clarification over the turns retained before this request.
        yield PipelineEvent::StageStarted {
            stage: ProgressStage::Clarifying,
        };
        let clarifier = Clarifier::with_model(Arc::clone(&model));
        let dialogue = clarifier
            .clarify_rounds(&query, &prior, &ClarifyConfig::new(DEFAULT_MAX_CLARIFY_ROUNDS))
            .await?;
        for round in &dialogue.rounds {
            yield PipelineEvent::Clarified(round.clone());
        }
        let (source, detail) = last_provenance(&dialogue.provenance);
        provenance.extend_from(dialogue.provenance);
        yield PipelineEvent::StageCompleted {
            stage: ProgressStage::Clarifying,
            source,
            detail,
        };
        if !dialogue.clarification.ready {
            let text = clarification_text(dialogue.clarification);
            provenance.warn_if_degraded();
            yield PipelineEvent::Final(build_result(
                &session_id,
                agent_type,
                BuildOutput::Response(text),
                BuildState::Clarifying,
                &provenance,
                &repair,
            ));
            return;
        }

        match agent_type {
            AgentType::LlmAgent => {
                let mut history = prior.clone();
                history.add_user_message(&query);
                yield PipelineEvent::StageStarted {
                    stage: ProgressStage::GeneratingConfig,
                };
                let (dsl, source, detail) =
                    llm_agent_pipeline(&model, &query, &history, timestamp_ms).await?;
                provenance.record(ProgressStage::GeneratingConfig, source, detail.clone());
                yield PipelineEvent::StageCompleted {
                    stage: ProgressStage::GeneratingConfig,
                    source,
                    detail,
                };
                provenance.warn_if_degraded();
                yield PipelineEvent::Final(build_result(
                    &session_id,
                    agent_type,
                    BuildOutput::Dsl(dsl),
                    BuildState::Completed,
                    &provenance,
                    &repair,
                ));
            }
            AgentType::Workflow => {
                // 4a. The three designer passes, reported one by one.
                let designer = WorkflowDesigner::with_model(Arc::clone(&model));
                let mut designs = designer.stream_design(&query);
                let mut design: Option<WorkflowDesign> = None;
                while let Some(event) = designs.next().await {
                    match event? {
                        DesignEvent::PassStarted(kind) => {
                            yield PipelineEvent::StageStarted {
                                stage: pass_stage(kind),
                            };
                        }
                        DesignEvent::PassCompleted(pass) => {
                            yield PipelineEvent::StageCompleted {
                                stage: pass_stage(pass.kind),
                                source: pass.source,
                                detail: pass.detail,
                            };
                        }
                        DesignEvent::Completed(three) => {
                            provenance.extend_from(three.provenance);
                            design = Some(three.design);
                        }
                    }
                }
                let Some(design) = design else {
                    Err(Error::validation(
                        "the designer stream ended without a design",
                    ))?;
                    return;
                };

                // 4b. Generate the workflow document.
                yield PipelineEvent::StageStarted {
                    stage: ProgressStage::GeneratingDl,
                };
                let generator = DlGenerator::with_model(Arc::clone(&model));
                let (mut current, source, detail) =
                    generate_workflow(&generator, &design).await?;
                provenance.record(ProgressStage::GeneratingDl, source, detail.clone());
                yield PipelineEvent::StageCompleted {
                    stage: ProgressStage::GeneratingDl,
                    source,
                    detail,
                };

                // 4c. The bounded repair loop: reflect, refine, reflect again.
                let reflector = DlReflector::new();
                let workflow = loop {
                    yield PipelineEvent::StageStarted {
                        stage: ProgressStage::ValidatingDl,
                    };
                    repair.record_attempt();
                    let reflection = match reflector.reflect(&current) {
                        Ok(reflection) => reflection,
                        Err(error) => {
                            tracing::debug!(
                                %error,
                                "deterministic reflection failed; keeping the generated document"
                            );
                            yield PipelineEvent::StageCompleted {
                                stage: ProgressStage::ValidatingDl,
                                source: GenerationSource::Deterministic,
                                detail: format!("reflection failed: {error}"),
                            };
                            break current;
                        }
                    };
                    let passed = reflection.passed;
                    let issue_count = reflection.issues.len();
                    let repaired = reflection.repaired;
                    yield PipelineEvent::StageCompleted {
                        stage: ProgressStage::ValidatingDl,
                        source: GenerationSource::Deterministic,
                        detail: if passed {
                            "no repair needed".to_string()
                        } else {
                            format!("{issue_count} issue(s) repaired")
                        },
                    };
                    yield PipelineEvent::Repaired(repair.clone());
                    if passed || repair.attempts >= DEFAULT_MAX_RETRIES {
                        break repaired;
                    }
                    repair.record_note(joined_issue_notes(&reflection.issues));
                    let feedback = repair_feedback(&reflection.issues);
                    yield PipelineEvent::StageStarted {
                        stage: ProgressStage::RefiningDl,
                    };
                    let refinement = match generator
                        .refine(&feedback, &repaired, &render_mermaid(&repaired))
                        .await
                    {
                        Ok(refinement) => refinement,
                        Err(error) => {
                            tracing::debug!(
                                %error,
                                "workflow refinement failed; keeping the repaired document"
                            );
                            yield PipelineEvent::StageCompleted {
                                stage: ProgressStage::RefiningDl,
                                source: GenerationSource::Deterministic,
                                detail: format!("refinement failed: {error}"),
                            };
                            break repaired;
                        }
                    };
                    let (source, detail) = last_provenance(&refinement.provenance);
                    provenance.extend_from(refinement.provenance);
                    yield PipelineEvent::StageCompleted {
                        stage: ProgressStage::RefiningDl,
                        source,
                        detail,
                    };
                    if refinement.workflow == repaired
                        || refinement.workflow.validate().is_err()
                    {
                        break repaired;
                    }
                    current = refinement.workflow;
                };
                provenance.warn_if_degraded();
                yield PipelineEvent::Final(build_result(
                    &session_id,
                    agent_type,
                    BuildOutput::Workflow(workflow),
                    BuildState::Completed,
                    &provenance,
                    &repair,
                ));
            }
        }
    })
}

/// The result one finished pipeline emits.
fn build_result(
    session_id: &str,
    agent_type: AgentType,
    output: BuildOutput,
    state: BuildState,
    provenance: &GenerationProvenance,
    repair: &RepairReport,
) -> BuildResult {
    BuildResult {
        status: map_state_to_status(state, agent_type),
        session_id: session_id.to_string(),
        agent_type,
        output,
        provenance: provenance.clone(),
        repair: repair.clone(),
    }
}

/// The progress stage a designer pass belongs to.
fn pass_stage(kind: DesignPassKind) -> ProgressStage {
    match kind {
        DesignPassKind::Basic | DesignPassKind::Branch => ProgressStage::GeneratingWorkflowDesign,
        DesignPassKind::Reflection => ProgressStage::ValidatingDl,
    }
}
