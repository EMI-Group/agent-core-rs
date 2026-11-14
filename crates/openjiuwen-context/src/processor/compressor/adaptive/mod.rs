//! Adaptive compression strategy selection.
//!
//! Ports the model-chosen branch of Python's compressor family. A compressor
//! that enables the flag asks one bounded, non-streaming model call which pass
//! the history actually needs, and receives the durable task state of the span
//! as a [`MemoryBlockPlan`] in the same reply.
//!
//! Two paths always exist:
//!
//! - **Model** ([`StrategySource::Model`]) — the configured [`AdaptiveStrategy`]
//!   has an enabled flag and a [`Model`]; its reply names a
//!   [`CompressionStrategy`] plus the plan. The call is bounded by
//!   [`AdaptiveStrategyConfig::max_decision_tokens`], never streams, and is
//!   refused outside the allowlist.
//! - **Deterministic fallback** ([`StrategySource::DeterministicFallback`]) —
//!   no model, a disabled flag, a history too small to be worth a call, a
//!   provider error, an empty or unparsable reply, or an unknown strategy name.
//!   [`AdaptiveStrategy::deterministic`] measures the history and walks a fixed
//!   ladder, so selection is total and works offline.
//!
//! The deterministic ladder, first match wins:
//!
//! 1. no measurable history → [`CompressionStrategy::Noop`];
//! 2. stale tool results hold at least `min_savings_ratio` of the tokens →
//!    [`CompressionStrategy::MicroCompact`] (a tool's older results can be
//!    cleared in place);
//! 3. an assistant turn is in flight and that round holds at least the ratio →
//!    [`CompressionStrategy::CurrentRound`];
//! 4. at least three completed rounds hold at least the ratio →
//!    [`CompressionStrategy::RoundLevel`];
//! 5. at least one completed round holds at least the ratio →
//!    [`CompressionStrategy::DialogueSummary`];
//! 6. the history is over its budget → [`CompressionStrategy::FullCompact`];
//! 7. otherwise [`AdaptiveStrategyConfig::fallback_strategy`] (default
//!    [`CompressionStrategy::Noop`]) with a measured reason.
//!
//! `min_savings_ratio` is the fraction of the history's tokens a candidate span
//! must hold before rewriting it is considered worth a pass.
//!
//! # Compressor contract
//!
//! A compressor that enables [`AdaptiveConfig`] runs the selection after its own
//! trigger gate, and then:
//!
//! - [`CompressionStrategy::Noop`] — leave the history and the counters alone and
//!   report nothing;
//! - the compressor's own strategy — proceed with the normal span/summary logic,
//!   but build the block body from the [`MemoryBlockPlan`] (model-backed when the
//!   reply carried one, deterministic otherwise) instead of the plain summary;
//! - another strategy — return a [`ContextEvent`] whose `summary` records the
//!   recommendation and whose `messages_changed` is empty, and leave the history
//!   unchanged, so the processor the recommendation names can run instead.
//!
//! With [`AdaptiveConfig::enabled`] off, no selection happens at all: a
//! compressor behaves exactly as it did before this module existed.
//!
//! [`Model`]: openjiuwen_foundation::llm::Model
//! [`ContextEvent`]: crate::events::ContextEvent

pub mod plan;
pub mod prompt;

mod config;
mod render;
mod strategy;
#[cfg(test)]
mod tests;

pub use self::config::AdaptiveConfig;
pub use self::plan::MemoryBlockPlan;
pub use self::render::{render_memory_block_from_plan, render_memory_block_from_plan_with_marker};
pub use self::strategy::{
    AdaptiveContext, AdaptiveStrategy, AdaptiveStrategyConfig, CompressionStrategy,
    StrategyDecision, StrategyInput, StrategyParse, StrategySource,
};

pub(crate) use self::render::render_block;
pub(crate) use self::strategy::{select_for_processor, AdaptiveOutcome};
