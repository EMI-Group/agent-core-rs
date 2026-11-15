//! The streaming view of the three designer passes.
//!
//! [`WorkflowDesigner::stream_design`] runs the same pipeline as
//! [`WorkflowDesigner::design_three_pass`] and reports it as it happens: one
//! [`DesignEvent::PassStarted`] and one [`DesignEvent::PassCompleted`] per pass,
//! then one [`DesignEvent::Completed`] carrying the finished
//! [`ThreePassDesign`].
//!
//! The stream owns a clone of the designer and of the request, so it is
//! `Send + 'static` and outlives both; nothing blocks, and every await is the
//! pass's own model round trip.

use futures::stream::BoxStream;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::super::outcome::GenerationProvenance;
use super::super::progress::ProgressStage;
use super::passes::{self, DesignPass, DesignPassKind, ThreePassDesign};
use super::WorkflowDesigner;

/// One event of the streamed three-pass designer pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DesignEvent {
    /// A pass is about to run.
    PassStarted(DesignPassKind),
    /// A pass finished, together with its record.
    PassCompleted(DesignPass),
    /// The pipeline finished; the payload is the same result
    /// [`WorkflowDesigner::design_three_pass`] returns.
    Completed(ThreePassDesign),
}

/// Streams the three passes for `query` over an owned `designer`.
pub(super) fn design_stream(
    designer: WorkflowDesigner,
    query: String,
) -> BoxStream<'static, Result<DesignEvent>> {
    Box::pin(async_stream::try_stream! {
        if query.trim().is_empty() {
            Err(Error::validation("query must not be blank"))?;
        }

        let mut provenance = GenerationProvenance::new();
        let mut records = Vec::with_capacity(3);

        yield DesignEvent::PassStarted(DesignPassKind::Basic);
        let basic = passes::run_basic(&designer, &query).await?;
        provenance.record(
            ProgressStage::GeneratingWorkflowDesign,
            basic.pass.source,
            basic.pass.detail.clone(),
        );
        yield DesignEvent::PassCompleted(basic.pass.clone());
        records.push(basic.pass);

        yield DesignEvent::PassStarted(DesignPassKind::Branch);
        let branch = passes::run_branch(&designer, &query, &basic.design).await?;
        provenance.record(
            ProgressStage::GeneratingWorkflowDesign,
            branch.pass.source,
            branch.pass.detail.clone(),
        );
        yield DesignEvent::PassCompleted(branch.pass.clone());
        records.push(branch.pass);

        yield DesignEvent::PassStarted(DesignPassKind::Reflection);
        let reflection = passes::run_reflection(&designer, &query, &branch.design).await?;
        provenance.record(
            ProgressStage::ValidatingDl,
            reflection.pass.source,
            reflection.pass.detail.clone(),
        );
        yield DesignEvent::PassCompleted(reflection.pass.clone());
        records.push(reflection.pass);

        provenance.warn_if_degraded();
        yield DesignEvent::Completed(ThreePassDesign {
            design: reflection.design,
            passes: records,
            provenance,
            notes: reflection.notes,
        });
    })
}

#[cfg(test)]
mod tests;
