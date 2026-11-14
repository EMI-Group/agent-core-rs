//! Rendering what a decision produces: the memory block a compressor writes and
//! the event that recommends another pass.
//!
//! The block keeps the compressor family's envelope and takes its body from the
//! [`MemoryBlockPlan`] the decision carried, so a model-chosen pass and the plain
//! path write the same shape. The recommendation event carries the advice and
//! changes no message.

use openjiuwen_foundation::prompt::BaseMessage;

use super::config::AdaptiveConfig;
use super::plan::MemoryBlockPlan;
use super::strategy::StrategyDecision;
use crate::events::{ContextEvent, ContextEventKind};
use crate::processor::compressor::{MemoryBlockKind, RecallArchiveMarker};

/// Renders `plan` inside `kind`'s memory-block envelope.
///
/// The envelope is unchanged (marker line, authority/instruction-status/
/// conflict-priority lines, `Summary:` label); the plan supplies the body: the
/// summary first, then the plan's structured sections. An empty plan renders the
/// same envelope a plain summary does.
#[must_use]
pub fn render_memory_block_from_plan(kind: MemoryBlockKind, plan: &MemoryBlockPlan) -> String {
    render_memory_block_from_plan_with_marker(kind, plan, None)
}

/// Renders `plan` inside `kind`'s envelope and appends a recall marker.
///
/// The marker is appended through [`RecallArchiveMarker::append`], the same
/// helper the non-adaptive path uses, so a block produced here points at its
/// recall archive exactly like a plain block.
#[must_use]
pub fn render_memory_block_from_plan_with_marker(
    kind: MemoryBlockKind,
    plan: &MemoryBlockPlan,
    marker: Option<&RecallArchiveMarker>,
) -> String {
    let block = kind.render(&plan.render_body());
    match marker {
        Some(marker) => marker.append(block),
        None => block,
    }
}

/// Renders the memory block a compressor writes, adaptive or not.
///
/// This is the one place that decides what a compressed block says:
///
/// - with [`AdaptiveConfig::enabled`] off the body is exactly
///   [`MemoryBlockKind::render`]'s plain `Summary:` text, so the non-adaptive
///   path stays byte-identical;
/// - otherwise the plan is the decision's model plan, gaps are filled from the
///   span when [`AdaptiveConfig::task_extraction`] is set, the summary is the
///   span summary the compressor computed, and the structured sections are
///   dropped when [`AdaptiveConfig::json_schema_blocks`] is off.
///
/// The recall marker, when the archive produced one, is appended through the
/// same [`RecallArchiveMarker::append`] helper the plain path uses.
pub(crate) fn render_block(
    kind: MemoryBlockKind,
    config: &AdaptiveConfig,
    decision: Option<&StrategyDecision>,
    span: &[BaseMessage],
    summary: &str,
    marker: Option<&RecallArchiveMarker>,
) -> String {
    let mut plan = if config.enabled {
        decision
            .and_then(|decision| decision.plan.clone())
            .unwrap_or_default()
    } else {
        MemoryBlockPlan::default()
    };
    if config.enabled && config.task_extraction {
        plan.fill_missing(&MemoryBlockPlan::from_messages(span, true));
    }
    summary.trim().clone_into(&mut plan.summary);
    if !config.enabled || !config.json_schema_blocks {
        plan = MemoryBlockPlan {
            summary: plan.summary,
            ..MemoryBlockPlan::default()
        };
    }
    render_memory_block_from_plan_with_marker(kind, &plan, marker)
}

/// The event a compressor reports when the model recommends another pass.
///
/// The event carries the recommendation in `summary` and changes no message, so
/// the history stays untouched. `ContextEngine::compress` treats a non-empty
/// event list as a report and may call the pass `compressed`; its token counts
/// are equal, which is how a caller recognises a recommendation.
pub(crate) fn recommendation_event(
    processor: &str,
    decision: &StrategyDecision,
    tokens: usize,
) -> ContextEvent {
    ContextEvent {
        processor: processor.to_owned(),
        kind: ContextEventKind::Summarized,
        messages_changed: Vec::new(),
        summary: Some(format!(
            "adaptive strategy recommendation: `{}` ({})",
            decision.strategy, decision.reason
        )),
        offload_handle: None,
        tokens_before: tokens,
        tokens_after: tokens,
    }
}
