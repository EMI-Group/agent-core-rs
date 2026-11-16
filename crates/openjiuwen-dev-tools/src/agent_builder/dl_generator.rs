//! The DL generator stage: a workflow design becomes a workflow DSL document.
//!
//! Ports `openjiuwen.dev_tools.agent_builder.builders.workflow.dl_generator`.
//! Python fills a large prompt template with component lists, node schemas, and
//! few-shot examples, then asks a model for a JSON workflow document. This port
//! keeps the observable artifact — a valid [`WorkflowDsl`] — and produces it
//! deterministically through [`WorkflowDesign::to_workflow_dsl`], with one
//! optional model round trip that may replace the deterministic document when
//! the answer parses and validates.
//!
//! Model problems are best-effort: a failed call, a choice-less answer, a
//! missing JSON fence, an unparseable document, or a document failing
//! [`WorkflowDsl::validate`] is logged and replaced by the deterministic
//! baseline, so [`DlGenerator::generate_with_model`] never fails because of the
//! model. Only an invalid *design* is a [`Error::Validation`].
//!
//! # Candidate mode
//!
//! [`DlGenerator::generate_candidate`] is the repair-aware form of the model
//! path: the model's parsed answer comes back even when it fails
//! [`WorkflowDsl::validate`], together with one note per structural failure, so
//! the pipeline's bounded repair loop can refine that document instead of
//! throwing the model's content away.
//! The deterministic document always comes back beside the candidate, so a caller that cannot repair the answer still has its baseline to fall back to.
//! The generation stage is recorded as `GenerationSource::Fallback` in that case, because its document is rejected; the repair loop may still rescue the answer, which then shows up as a later `refining_dl` = `model` record.
//!
//! # Refine mode
//!
//! [`DlGenerator::refine`] ports Python's `DlGenerator.refine`: instead of a
//! design it takes the requested change, the existing workflow document, and the
//! document's Mermaid diagram, and asks the model for the complete modified
//! document. The answer is accepted only when it parses and validates;
//! otherwise — and when no model is configured at all — the existing document is
//! returned unchanged. Every call reports where its result came from in a
//! [`GenerationProvenance`] holding one `refining_dl` record.

use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::BaseMessage;
use serde::{Deserialize, Serialize};

use super::designer::WorkflowDesign;
use super::outcome::GenerationProvenance;
use super::progress::ProgressStage;
use super::spec::WorkflowSpec;
use super::utils::extract_json_from_text;
use super::workflow_dsl::WorkflowDsl;
use crate::util::extract_json_fence;

/// The instruction sent to the model by the generation round trip of
/// [`DlGenerator::generate_with_model`] and [`DlGenerator::generate_candidate`].
///
/// It names every key the parser reads, so a compliant answer round-trips into
/// [`WorkflowDsl`] without repair.
const DL_GENERATOR_SYSTEM_PROMPT: &str = r#"You are the openJiuwen workflow DL generator.
Turn the given workflow design into one complete workflow DSL document.

Answer with exactly one ```json fenced code block and nothing else inside it,
using these keys:
- `name`: the workflow name
- `description`: one sentence describing the workflow
- `nodes`: a list of {"id", "type", "title", "description", "inputs",
  "outputs", "data"} objects, keeping the design's node ids
- `edges`: a list of {"source", "target", "condition"} objects naming node ids

The document needs at least one node, unique node ids, and every edge endpoint
must resolve to a node id.
Do not add other keys and do not wrap the document in prose."#;

/// The instruction sent to the model by [`DlGenerator::refine`].
///
/// It is Python's refine system prompt: the same generator contract, applied to
/// an existing document instead of a design.
const DL_REFINE_SYSTEM_PROMPT: &str = r#"You are the openJiuwen workflow DL generator in refinement mode.
Rewrite the existing workflow DSL document the user sends so that it satisfies
the requested change and stays free of errors.

Answer with exactly one ```json fenced code block holding the complete modified
workflow document, using these keys:
- `name`: the workflow name
- `description`: one sentence describing the workflow
- `nodes`: a list of {"id", "type", "title", "description", "inputs",
  "outputs", "data"} objects, keeping the existing node ids where possible
- `edges`: a list of {"source", "target", "condition"} objects naming node ids

The document needs at least one node, unique node ids, and every edge endpoint
must resolve to a node id.
Do not add other keys and do not wrap the document in prose."#;

/// The user template of Python's refine prompt, filled by
/// [`refine_user_message`].
///
/// The markers are the three values Python's `DL_REFINE_USER_TEMPLATE` formats:
/// the requested change, the existing Mermaid diagram, and the existing
/// workflow document.
const DL_REFINE_USER_PROMPT_TEMPLATE: &str = r"Refine the existing workflow according to the requested change.
Make sure the result satisfies the request and is free of errors.

## Requested change
{feedback}

## Existing Mermaid diagram
{mermaid}

## Existing workflow document
{dsl}";

/// One workflow document produced by the generation stage, before repair.
///
/// The candidate is the model's answer as answered — including a document that
/// fails [`WorkflowDsl::validate`], with one [`Self::issues`] note per structural
/// failure — so a repair loop can refine it. The deterministic document is
/// always present beside it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GeneratedDl {
    /// The model's parsed answer, present even when it fails
    /// [`WorkflowDsl::validate`]; `None` when no model is configured or the
    /// model could not produce a document.
    pub candidate: Option<WorkflowDsl>,
    /// The deterministic document for the same design.
    pub baseline: WorkflowDsl,
    /// One note per structural failure of `candidate`; empty when it validated
    /// or is absent.
    pub issues: Vec<String>,
    /// Exactly one [`ProgressStage::GeneratingDl`] record describing where the
    /// generation answer came from.
    pub provenance: GenerationProvenance,
}

impl GeneratedDl {
    /// The document to work with: the model's candidate when it answered,
    /// otherwise the deterministic baseline.
    #[must_use]
    pub fn start(&self) -> &WorkflowDsl {
        self.candidate.as_ref().unwrap_or(&self.baseline)
    }

    /// Whether the model answered with a document at all, valid or not.
    #[must_use]
    pub fn is_model_answer(&self) -> bool {
        self.candidate.is_some()
    }

    /// Whether the model's document parsed and validated as answered.
    #[must_use]
    pub fn is_accepted(&self) -> bool {
        self.candidate.is_some() && self.issues.is_empty()
    }
}

/// One model-assisted refinement of an existing workflow document.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Refinement {
    /// The accepted revision, or the caller's document unchanged on a fallback.
    pub workflow: WorkflowDsl,
    /// One `refining_dl` record naming where `workflow` came from.
    pub provenance: GenerationProvenance,
}

/// Generates a workflow DSL document from a workflow design.
///
/// [`Self::generate`] is deterministic and needs no model; [`Self::with_model`]
/// adds one best-effort model round trip behind
/// [`Self::generate_with_model`], and [`Self::from_spec`] converts a
/// programmatic specification without ever calling a model.
pub struct DlGenerator {
    /// The optional model the model path talks to.
    model: Option<Arc<dyn Model>>,
}

impl DlGenerator {
    /// Creates a deterministic generator; the model path never calls a model.
    #[must_use]
    pub fn new() -> Self {
        Self { model: None }
    }

    /// Creates a generator whose [`Self::generate_with_model`] asks `model`
    /// first and falls back to the deterministic document.
    #[must_use]
    pub fn with_model(model: Arc<dyn Model>) -> Self {
        Self { model: Some(model) }
    }

    /// Converts a design into a validated workflow DSL document.
    ///
    /// The conversion is the one design-to-DSL mapping of the crate
    /// ([`WorkflowDesign::to_workflow_dsl`]), which validates both the design
    /// and the produced document.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the design or the produced document
    /// is structurally invalid.
    pub fn generate(&self, design: &WorkflowDesign) -> Result<WorkflowDsl> {
        design.to_workflow_dsl()
    }

    /// Converts a specification into a validated workflow DSL document.
    ///
    /// The name, description, nodes, and edges are kept unchanged; the document
    /// is validated before it is returned. No model is ever called.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the document has no nodes, repeated
    /// node ids, or an edge endpoint that does not name a node.
    pub fn from_spec(&self, spec: &WorkflowSpec) -> Result<WorkflowDsl> {
        let document = spec.to_dsl();
        document.validate()?;
        Ok(document)
    }

    /// Converts a design into a workflow DSL document, using the model when one
    /// is configured.
    ///
    /// The model's document is used only when it parses *and*
    /// [`WorkflowDsl::validate`] passes; every other outcome — no model, a model
    /// error, a choice-less answer, a missing JSON fence, an unparseable
    /// document, or an invalid one — yields the deterministic baseline. That is
    /// exactly [`Self::generate_candidate`] with the candidate dropped unless it
    /// validated, so an invalid design is still a validation error before any
    /// model call and the model is still called at most once.
    ///
    /// # Errors
    ///
    /// Returns the [`Error::Validation`] of [`Self::generate`] for an invalid
    /// design — the only error this entry point produces.
    pub async fn generate_with_model(&self, design: &WorkflowDesign) -> Result<WorkflowDsl> {
        let GeneratedDl {
            candidate,
            baseline,
            issues,
            ..
        } = self.generate_candidate(design).await?;
        match candidate {
            Some(document) if issues.is_empty() => Ok(document),
            _ => Ok(baseline),
        }
    }

    /// Converts a design into a workflow-DSL candidate for the repair loop.
    ///
    /// The deterministic baseline is computed first, so an invalid design is a
    /// validation error regardless of the model.
    /// Without a model the result holds no candidate and one
    /// [`GenerationSource::Deterministic`] record.
    /// Otherwise the single model round trip produces the document the model
    /// answered with — *even when it fails [`WorkflowDsl::validate`]* — beside
    /// one note per structural failure, so a caller can repair it instead of
    /// keeping the baseline.
    /// A parsed answer that fails [`WorkflowDsl::validate`] is recorded as
    /// [`GenerationSource::Fallback`], because its document is rejected and the
    /// stage delivered no usable document; the candidate is still handed over
    /// for repair.
    /// The repair loop may still rescue such an answer, which then shows up as a
    /// later `refining_dl` = `model` record.
    /// A failed call, a choice-less answer, a missing JSON fence, or an
    /// unparseable document yields no candidate, no issues, and one
    /// [`GenerationSource::Fallback`] record naming the reason.
    ///
    /// Degradation is not warned about here: the pipeline warns once at the end
    /// of a build.
    ///
    /// [`GenerationSource::Deterministic`]: super::outcome::GenerationSource::Deterministic
    /// [`GenerationSource::Fallback`]: super::outcome::GenerationSource::Fallback
    ///
    /// # Errors
    ///
    /// Returns the [`Error::Validation`] of [`Self::generate`] for an invalid
    /// design — the only error this entry point produces.
    pub async fn generate_candidate(&self, design: &WorkflowDesign) -> Result<GeneratedDl> {
        let baseline = self.generate(design)?;
        let mut provenance = GenerationProvenance::new();
        let Some(model) = self.model.as_deref() else {
            provenance.record_deterministic(
                ProgressStage::GeneratingDl,
                "no model configured; using the deterministic document",
            );
            return Ok(GeneratedDl {
                candidate: None,
                baseline,
                issues: Vec::new(),
                provenance,
            });
        };

        match request_document(model, design).await {
            Ok(document) => {
                let issues: Vec<String> = document
                    .validate()
                    .err()
                    .map(|error| vec![error.to_string()])
                    .unwrap_or_default();
                if issues.is_empty() {
                    provenance.record_model(ProgressStage::GeneratingDl);
                } else {
                    provenance.record_fallback(
                        ProgressStage::GeneratingDl,
                        format!("model document needs repair: {}", issues.join("; ")),
                    );
                }
                Ok(GeneratedDl {
                    candidate: Some(document),
                    baseline,
                    issues,
                    provenance,
                })
            }
            Err(reason) => {
                tracing::debug!(
                    %reason,
                    "workflow DL generation produced no document; using the deterministic document"
                );
                provenance.record_fallback(ProgressStage::GeneratingDl, reason);
                Ok(GeneratedDl {
                    candidate: None,
                    baseline,
                    issues: Vec::new(),
                    provenance,
                })
            }
        }
    }

    /// Rewrites `existing` according to `feedback`, using Python's refine prompt.
    ///
    /// The model receives the refine system contract, then one user message
    /// carrying `feedback`, the `mermaid` diagram of the current document, and the
    /// document itself as JSON. Its answer is accepted only when a fenced-or-bare
    /// JSON document parses as a [`WorkflowDsl`] and
    /// [`WorkflowDsl::validate`] passes.
    ///
    /// Without a model the existing document is returned unchanged with a
    /// [`GenerationSource::Deterministic`] record. A model error, a choice-less
    /// answer, an unparseable answer, or an invalid document is logged and also
    /// keeps the existing document, recorded as [`GenerationSource::Fallback`] with
    /// the reason as its detail. The returned provenance always holds exactly one
    /// [`ProgressStage::RefiningDl`] record.
    ///
    /// [`GenerationSource::Deterministic`]: super::outcome::GenerationSource::Deterministic
    /// [`GenerationSource::Fallback`]: super::outcome::GenerationSource::Fallback
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `feedback` is blank; the model is never
    /// called in that case.
    pub async fn refine(
        &self,
        feedback: &str,
        existing: &WorkflowDsl,
        mermaid: &str,
    ) -> Result<Refinement> {
        if feedback.trim().is_empty() {
            return Err(Error::validation("refinement feedback must not be blank"));
        }

        let mut provenance = GenerationProvenance::new();
        let Some(model) = self.model.as_deref() else {
            provenance.record_deterministic(
                ProgressStage::RefiningDl,
                "no model configured; keeping the existing workflow",
            );
            provenance.warn_if_degraded();
            return Ok(Refinement {
                workflow: existing.clone(),
                provenance,
            });
        };

        match refine_with_model(model, feedback, existing, mermaid).await {
            Ok(workflow) => {
                provenance.record_model(ProgressStage::RefiningDl);
                Ok(Refinement {
                    workflow,
                    provenance,
                })
            }
            Err(reason) => {
                tracing::debug!(%reason, "workflow DL refinement failed; keeping the existing document");
                provenance.record_fallback(ProgressStage::RefiningDl, reason);
                provenance.warn_if_degraded();
                Ok(Refinement {
                    workflow: existing.clone(),
                    provenance,
                })
            }
        }
    }
}

impl Default for DlGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for DlGenerator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DlGenerator")
            .field("has_model", &self.model.is_some())
            .finish()
    }
}

/// Asks the model for one fenced workflow document, without validating it.
///
/// The failure is a short reason rather than an error, because every generation
/// failure keeps the deterministic baseline and is reported as provenance
/// detail. Validation is the caller's job: [`DlGenerator::generate_candidate`]
/// keeps an invalid document and reports what is wrong with it.
async fn request_document(
    model: &dyn Model,
    design: &WorkflowDesign,
) -> std::result::Result<WorkflowDsl, String> {
    let design_json =
        serde_json::to_string_pretty(design).unwrap_or_else(|_| format!("{design:?}"));
    let messages = vec![
        BaseMessage::system(DL_GENERATOR_SYSTEM_PROMPT),
        BaseMessage::user(design_json),
    ];
    let request = ModelRequestConfig::new(model.model_name());
    let response = model
        .invoke(&messages, &request)
        .await
        .map_err(|error| format!("model call failed: {error}"))?;
    let Some(choice) = response.choices.into_iter().next() else {
        return Err("model returned no choices".to_string());
    };

    let text = choice.message.text().into_owned();
    let value = extract_json_fence(&text)
        .ok_or_else(|| "model answer contained no JSON code fence".to_string())?;
    serde_json::from_value(value)
        .map_err(|error| format!("failed to parse workflow DSL JSON: {error}"))
}

/// Asks the model for one refined workflow document, or reports why it failed.
///
/// The failure is a short reason rather than an error, because every refine
/// failure keeps the caller's document and is recorded as provenance detail.
async fn refine_with_model(
    model: &dyn Model,
    feedback: &str,
    existing: &WorkflowDsl,
    mermaid: &str,
) -> std::result::Result<WorkflowDsl, String> {
    let document_json =
        serde_json::to_string_pretty(existing).unwrap_or_else(|_| format!("{existing:?}"));
    let messages = vec![
        BaseMessage::system(DL_REFINE_SYSTEM_PROMPT),
        BaseMessage::user(refine_user_message(feedback, mermaid, &document_json)),
    ];
    let request = ModelRequestConfig::new(model.model_name());
    let response = model
        .invoke(&messages, &request)
        .await
        .map_err(|error| format!("model call failed: {error}"))?;
    let Some(choice) = response.choices.into_iter().next() else {
        return Err("model returned no choices".to_string());
    };

    let text = choice.message.text().into_owned();
    let json = extract_json_from_text(&text);
    let document: WorkflowDsl = serde_json::from_str(json.trim())
        .map_err(|error| format!("answer is not a workflow document: {error}"))?;
    document
        .validate()
        .map_err(|error| format!("revised document is invalid: {error}"))?;
    Ok(document)
}

/// Fills [`DL_REFINE_USER_PROMPT_TEMPLATE`] from its three markers.
///
/// The template is split on each marker once, in template order, so braces inside
/// the inserted feedback, diagram, or document are never reinterpreted.
fn refine_user_message(feedback: &str, mermaid: &str, dsl: &str) -> String {
    let Some((head, rest)) = DL_REFINE_USER_PROMPT_TEMPLATE.split_once("{feedback}") else {
        return DL_REFINE_USER_PROMPT_TEMPLATE.to_string();
    };
    let Some((after_feedback, rest)) = rest.split_once("{mermaid}") else {
        return DL_REFINE_USER_PROMPT_TEMPLATE.to_string();
    };
    let Some((after_mermaid, tail)) = rest.split_once("{dsl}") else {
        return DL_REFINE_USER_PROMPT_TEMPLATE.to_string();
    };
    format!("{head}{feedback}{after_feedback}{mermaid}{after_mermaid}{dsl}{tail}")
}

#[cfg(test)]
#[path = "dl_generator/tests.rs"]
mod tests;
