//! The serde-facing adaptive sub-config a compressor carries.
//!
//! [`AdaptiveConfig`] is the serializable half of the adaptive seam: the knob a
//! compressor config exposes, so a configuration round-trips through JSON. It
//! carries no model (that comes from the compressor's own field) and can build
//! the full [`AdaptiveStrategyConfig`] or the [`AdaptiveStrategy`] the selection
//! runs on.

use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;
use serde::{Deserialize, Serialize};

use super::strategy::{AdaptiveStrategy, AdaptiveStrategyConfig, CompressionStrategy};

/// The compressing configs' serde-facing adaptive sub-config.
///
/// It carries only the serializable knobs — the model comes from the
/// compressor's own `Option<Arc<dyn Model>>` — so a compressor config round-trips
/// through JSON unchanged. [`AdaptiveConfig::enabled`] defaults to `false`, which
/// keeps every existing compressor byte-identical.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AdaptiveConfig {
    /// Whether this compressor runs adaptive selection before compressing.
    pub enabled: bool,
    /// Whether the deterministic extractor fills gaps in the plan.
    pub task_extraction: bool,
    /// Whether rendered blocks carry the structured plan sections.
    pub json_schema_blocks: bool,
    /// Upper bound on the decision reply's tokens.
    pub max_decision_tokens: usize,
    /// Strategies the model may choose; empty allows every strategy.
    pub strategy_allowlist: Vec<CompressionStrategy>,
    /// Strategy the ladder ends on when no measured span dominates.
    pub fallback_strategy: CompressionStrategy,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            task_extraction: true,
            json_schema_blocks: true,
            max_decision_tokens: 1_024,
            strategy_allowlist: Vec::new(),
            fallback_strategy: CompressionStrategy::Noop,
        }
    }
}

impl AdaptiveConfig {
    /// Checks the adaptive configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](openjiuwen_common::error::Error::Config) when
    /// the flag is on with a zero decision budget
    /// or a savings ratio outside `(0, 1]`. A disabled config never fails, so
    /// existing configurations keep validating.
    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        self.strategy_config().validate()
    }

    /// The full strategy configuration this sub-config describes.
    #[must_use]
    pub fn strategy_config(&self) -> AdaptiveStrategyConfig {
        AdaptiveStrategyConfig {
            enabled: self.enabled,
            max_decision_tokens: self.max_decision_tokens,
            strategy_allowlist: self.strategy_allowlist.clone(),
            task_extraction: self.task_extraction,
            json_schema_blocks: self.json_schema_blocks,
            fallback_strategy: self.fallback_strategy,
            ..AdaptiveStrategyConfig::default()
        }
    }

    /// Builds the selection seam, wiring `model` as the decision model.
    #[must_use]
    pub fn strategy(&self, model: Option<Arc<dyn Model>>) -> AdaptiveStrategy {
        AdaptiveStrategy::new(self.strategy_config()).with_model(model)
    }
}
