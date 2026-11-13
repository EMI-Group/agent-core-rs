//! Micro compact: clear the text of stale tool results.
//!
//! Ports `processor/compressor/micro_compact_processor.py`. Long agent runs
//! accumulate hundreds of tool results whose bodies stop being useful: the model
//! only needs the fact that the call happened. This processor keeps the newest
//! [`MicroCompactProcessorConfig::keep_recent_per_tool`] results of each
//! compactable tool verbatim and replaces the body of every older one with
//! [`MicroCompactProcessorConfig::cleared_marker`].
//!
//! Unlike the other compressor processors it never changes the structure of the
//! history: no message is removed or reordered, so every tool-call/tool-result
//! pair stays intact by construction. The role, `tool_call_id`, `name`,
//! `metadata`, and `tool_calls` of a cleared message are preserved — only the
//! text changes.
//!
//! The processor is fully deterministic and never calls a model. It runs only
//! when the history ends on a completed API round, so a pass never clears
//! results the model is still waiting for.
use std::collections::BTreeMap;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::{BaseMessage, MessageContent, Role};
use serde::{Deserialize, Serialize};

use crate::events::{ContextEvent, ContextEventKind};
use crate::processor::compressor::compact::ends_on_completed_api_round;
use crate::processor::offloader::resolve_tool_name;
use crate::processor::{ContextProcessor, ProcessorContext};
use crate::tokens::count_messages;

/// Replacement text of a cleared tool result.
pub const MICRO_COMPACT_CLEARED_MARKER: &str = "[Old tool result content cleared]";

/// Tool names whose older results are cleared by default.
fn default_compactable_tool_names() -> Vec<String> {
    ["grep", "glob", "read_file", "web_search", "web_fetch"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

/// Configuration of [`MicroCompactProcessor`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MicroCompactProcessorConfig {
    /// Clear stale results only after a tool keeps this many beyond the tail.
    pub trigger_threshold: usize,
    /// Tool names whose older result bodies may be cleared.
    pub compactable_tool_names: Vec<String>,
    /// Newest results preserved verbatim per compactable tool.
    pub keep_recent_per_tool: usize,
    /// Replacement text of a cleared result.
    pub cleared_marker: String,
}

impl Default for MicroCompactProcessorConfig {
    fn default() -> Self {
        Self {
            trigger_threshold: 5,
            compactable_tool_names: default_compactable_tool_names(),
            keep_recent_per_tool: 15,
            cleared_marker: MICRO_COMPACT_CLEARED_MARKER.to_owned(),
        }
    }
}

impl MicroCompactProcessorConfig {
    /// Checks the trigger threshold.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when
    /// [`trigger_threshold`](MicroCompactProcessorConfig::trigger_threshold) is
    /// zero, which would clear on every pass with no meaningful gate.
    pub fn validate(&self) -> Result<()> {
        if self.trigger_threshold == 0 {
            return Err(Error::Config(
                "micro compact: trigger_threshold must be greater than zero".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Clears the text of stale tool results, newest ones excepted.
#[derive(Debug, Clone)]
pub struct MicroCompactProcessor {
    /// Processor configuration.
    pub config: MicroCompactProcessorConfig,
}

impl MicroCompactProcessor {
    /// Creates a processor with `config`.
    #[must_use]
    pub fn new(config: MicroCompactProcessorConfig) -> Self {
        Self { config }
    }

    /// Whether this processor may clear results of `name`.
    fn is_compactable(&self, name: &str) -> bool {
        self.config
            .compactable_tool_names
            .iter()
            .any(|allowed| allowed == name)
    }

    /// Indices of the results to clear, grouped by tool name.
    fn collect_clearable(&self, messages: &[BaseMessage], manual: bool) -> Vec<usize> {
        let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (index, message) in messages.iter().enumerate() {
            if message.role != Role::Tool {
                continue;
            }
            if message.text().as_ref() == self.config.cleared_marker {
                continue;
            }
            let Some(name) = resolve_tool_name(messages, index) else {
                continue;
            };
            if !self.is_compactable(name) {
                continue;
            }
            groups.entry(name).or_default().push(index);
        }

        let threshold = if manual {
            self.config.keep_recent_per_tool
        } else {
            self.config
                .keep_recent_per_tool
                .saturating_add(self.config.trigger_threshold)
        };
        let mut clearable: Vec<usize> = Vec::new();
        for indices in groups.values() {
            if indices.len() <= threshold {
                continue;
            }
            let keep = self.config.keep_recent_per_tool.min(indices.len());
            clearable.extend(indices[..indices.len() - keep].iter().copied());
        }
        clearable.sort_unstable();
        clearable
    }
}

#[async_trait]
impl ContextProcessor for MicroCompactProcessor {
    fn name(&self) -> &'static str {
        "micro_compact"
    }

    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        if !ends_on_completed_api_round(messages) {
            return Ok(None);
        }
        if let Some(event) = self.clear(messages, cx) {
            return Ok(Some(event));
        }
        Ok(None)
    }
}

impl MicroCompactProcessor {
    /// Clears the bodies and reports the change, or `None` when nothing changed.
    fn clear(
        &self,
        messages: &mut [BaseMessage],
        cx: &ProcessorContext<'_>,
    ) -> Option<ContextEvent> {
        let clearable = self.collect_clearable(messages, cx.manual);
        if clearable.is_empty() {
            return None;
        }
        let tokens_before = count_messages(cx.counter, messages);
        for index in &clearable {
            let mut cleared = messages[*index].clone();
            cleared.content = MessageContent::Text(self.config.cleared_marker.clone());
            messages[*index] = cleared;
        }
        let tokens_after = count_messages(cx.counter, messages);
        Some(ContextEvent {
            processor: self.name().to_owned(),
            kind: ContextEventKind::ToolResultTrimmed,
            messages_changed: clearable,
            summary: None,
            offload_handle: None,
            tokens_before,
            tokens_after,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid_and_round_trips() {
        let config = MicroCompactProcessorConfig::default();
        config.validate().unwrap();
        assert_eq!(
            serde_json::from_str::<MicroCompactProcessorConfig>("{}").unwrap(),
            config
        );
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(
            serde_json::from_str::<MicroCompactProcessorConfig>(&json).unwrap(),
            config
        );
    }

    #[test]
    fn a_zero_threshold_is_rejected() {
        let config = MicroCompactProcessorConfig {
            trigger_threshold: 0,
            ..MicroCompactProcessorConfig::default()
        };
        assert!(matches!(config.validate(), Err(Error::Config(_))));
    }
}
