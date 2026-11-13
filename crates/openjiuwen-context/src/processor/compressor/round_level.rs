//! Round-level compressor: the last-resort fallback ladder.
//!
//! Ports the deterministic core of Python's
//! `processor/compressor/round_level_compressor.py`. When the dialogue and
//! current-round compressors are not enough, this compressor keeps applying
//! cheaper and more aggressive transforms until the history fits
//! [`RoundLevelCompressorConfig::target_total_tokens`]:
//!
//! 1. **Blocks** — every completed round in the compressible prefix becomes one
//!    [`ROUND_LEVEL_MEMORY_BLOCK`] user message.
//! 2. **Merge** — contiguous runs of round-level blocks are collapsed into one
//!    block, joining the bodies with `\n---\n`.
//! 3. **Aggressive** — the completed rounds phase 1 left behind — including the
//!    ones `keep_recent_messages` protected — are summarised again against a
//!    smaller target.
//! 4. **Hard truncation** — the remaining prefix collapses into one
//!    `truncated_full_context` block holding the head and tail of the offline
//!    transcript around the truncation marker.
//!
//! Every step is benefit-gated: a replacement is applied only when it is
//! strictly smaller than what it replaces, so the history never grows and a
//! summary that is longer than its source is rejected. System messages are never
//! removed, and only whole completed rounds are replaced, so a
//! tool-call/tool-result pair is never split.
//!
//! [`ROUND_LEVEL_MEMORY_BLOCK`]: crate::processor::compressor::ROUND_LEVEL_MEMORY_BLOCK
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{BaseMessage, Role, TokenCounter};
use serde::{Deserialize, Serialize};

use crate::events::{ContextEvent, ContextEventKind};
use crate::processor::compressor::{
    append_recall_markers, archive_replaced_messages, group_completed_rounds, is_memory_block,
    iter_summary_merge_ranges, rollback_recall_archive, split_recall_markers, summarize,
    summarize_offline, MemoryBlockKind, ROUND_LEVEL_MEMORY_BLOCK, SUMMARY_JOIN_SEPARATOR,
};
use crate::processor::{ContextProcessor, ProcessorContext};
use crate::tokens::{count_message, count_messages};

/// Characters per token the character budgets assume.
///
/// The configured budgets are token budgets; the offline transcript is sized in
/// characters with this ratio and then corrected against the real counter, so a
/// non-default counter still lands inside the token target.
const CHARS_PER_TOKEN_ESTIMATE: usize = 4;

/// Scope written into the block that replaced the whole compressible prefix.
pub const TRUNCATED_FULL_CONTEXT_SCOPE: &str = "truncated_full_context";

/// Instruction used for the block summaries.
const COMPRESSION_PROMPT: &str = "Summarise this conversation block into a compact \
reference-memory block; preserve identifiers, decisions, file paths and open questions; \
do not invent details";

/// Upper bound on the hard-truncation refinement rounds.
const MAX_TRUNCATION_ATTEMPTS: usize = 8;

/// Configuration of [`RoundLevelCompressor`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RoundLevelCompressorConfig {
    /// Context-window utilisation ratio that triggers the compressor.
    pub trigger_context_ratio: f32,
    /// Target size of the whole history after a pass.
    pub target_total_tokens: usize,
    /// Newest messages that phase 1 never compresses.
    pub keep_recent_messages: usize,
    /// Maximum size of one summary-generation call.
    pub compression_call_max_tokens: usize,
    /// Token target of a phase-1 block.
    pub first_pass_target_tokens: usize,
    /// Token target of a phase-3 block.
    pub second_pass_target_tokens: usize,
    /// Character cap of a phase-2 merged body.
    pub third_pass_target_tokens: usize,
    /// Fraction of the truncation budget kept from the head of the transcript.
    pub truncate_head_ratio: f32,
    /// Inserted between the head and the tail of a truncated transcript.
    pub truncated_marker: String,
    /// Marker that opens a round-level memory block.
    pub compression_marker: String,
}

impl Default for RoundLevelCompressorConfig {
    fn default() -> Self {
        Self {
            trigger_context_ratio: 0.9,
            target_total_tokens: 160_000,
            keep_recent_messages: 0,
            compression_call_max_tokens: 250_000,
            first_pass_target_tokens: 30_000,
            second_pass_target_tokens: 20_000,
            third_pass_target_tokens: 10_000,
            truncate_head_ratio: 0.2,
            truncated_marker: crate::processor::compressor::TRUNCATED_MARKER.to_owned(),
            compression_marker: ROUND_LEVEL_MEMORY_BLOCK.to_owned(),
        }
    }
}

impl RoundLevelCompressorConfig {
    /// Checks the ratios and the phase sizes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when
    /// [`trigger_context_ratio`](RoundLevelCompressorConfig::trigger_context_ratio)
    /// is outside `(0.0, 1.0]`, when
    /// [`truncate_head_ratio`](RoundLevelCompressorConfig::truncate_head_ratio) is
    /// outside `(0.0, 1.0)`, or when a phase size is zero.
    pub fn validate(&self) -> Result<()> {
        if !(self.trigger_context_ratio > 0.0 && self.trigger_context_ratio <= 1.0) {
            return Err(Error::Config(format!(
                "round-level compressor: trigger_context_ratio ({}) must be inside (0.0, 1.0]",
                self.trigger_context_ratio
            )));
        }
        if !(self.truncate_head_ratio > 0.0 && self.truncate_head_ratio < 1.0) {
            return Err(Error::Config(format!(
                "round-level compressor: truncate_head_ratio ({}) must be inside (0.0, 1.0)",
                self.truncate_head_ratio
            )));
        }
        for (name, value) in [
            ("target_total_tokens", self.target_total_tokens),
            (
                "compression_call_max_tokens",
                self.compression_call_max_tokens,
            ),
            ("first_pass_target_tokens", self.first_pass_target_tokens),
            ("second_pass_target_tokens", self.second_pass_target_tokens),
            ("third_pass_target_tokens", self.third_pass_target_tokens),
        ] {
            if value == 0 {
                return Err(Error::Config(format!(
                    "round-level compressor: {name} must be greater than zero"
                )));
            }
        }
        Ok(())
    }
}

/// Compresses a history down to a token target with a deterministic ladder.
#[derive(Clone)]
pub struct RoundLevelCompressor {
    /// Processor configuration.
    pub config: RoundLevelCompressorConfig,
    /// Model used when the request does not supply one.
    pub model: Option<Arc<dyn Model>>,
}

impl std::fmt::Debug for RoundLevelCompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoundLevelCompressor")
            .field("config", &self.config)
            .field("has_model", &self.model.is_some())
            .finish()
    }
}

/// The character budget equivalent of a token budget.
///
/// The ratio is measured through `counter`, so a one-character-per-token counter
/// and the default four-characters-per-token counter both get a budget of the
/// same token size.
fn chars_per_token(counter: &dyn TokenCounter) -> usize {
    const SAMPLE_CHARS: usize = 400;
    let sample = "x".repeat(SAMPLE_CHARS);
    let tokens = counter.count_text(&sample);
    SAMPLE_CHARS
        .checked_div(tokens)
        .unwrap_or(CHARS_PER_TOKEN_ESTIMATE)
        .max(1)
}

/// Keeps the head and the tail of `transcript` around `marker`.
///
/// `limit` is the character budget of the whole body, so the marker and the two
/// joining newlines are paid for first. A zero budget yields the marker alone; a
/// transcript that already fits is returned unchanged.
fn head_tail_body(transcript: &str, limit: usize, head_ratio: f32, marker: &str) -> String {
    let chars: Vec<char> = transcript.chars().collect();
    let marker_cost = marker.chars().count() + 2;
    if chars.len() + marker_cost <= limit {
        return transcript.to_owned();
    }
    let content = limit.saturating_sub(marker_cost);
    if content == 0 {
        return marker.to_owned();
    }
    let head_len = ((content as f32 * head_ratio) as usize).clamp(1, content);
    let tail_len = content - head_len;
    let head: String = chars[..head_len].iter().collect();
    let mut body = format!("{head}\n{marker}");
    if tail_len > 0 {
        let tail: String = chars[chars.len() - tail_len..].iter().collect();
        body.push('\n');
        body.push_str(&tail);
    }
    body
}

impl RoundLevelCompressor {
    /// Creates a compressor with `config` and an optional fallback `model`.
    #[must_use]
    pub fn new(config: RoundLevelCompressorConfig, model: Option<Arc<dyn Model>>) -> Self {
        Self { config, model }
    }

    /// The configured block marker.
    fn marker(&self) -> &str {
        &self.config.compression_marker
    }

    /// Renders one memory block, honouring a custom marker.
    fn render_block(&self, scope: &str, summary: &str) -> String {
        let rendered = MemoryBlockKind::RoundLevel.render_with_scope(summary, scope);
        if self.config.compression_marker == ROUND_LEVEL_MEMORY_BLOCK {
            return rendered;
        }
        match rendered.strip_prefix(ROUND_LEVEL_MEMORY_BLOCK) {
            Some(rest) => format!("{}{rest}", self.config.compression_marker),
            None => rendered,
        }
    }

    /// Whether `message` is a block this compressor wrote.
    fn is_block(&self, message: &BaseMessage) -> bool {
        is_memory_block(message, self.marker())
    }

    /// The body of a rendered block, tolerant of a custom marker.
    fn body(text: &str) -> &str {
        text.split_once("\nSummary:\n")
            .map_or(text, |(_, body)| body.trim())
    }

    /// Whether the thresholds ask for a pass.
    fn should_run(&self, total: usize, cx: &ProcessorContext<'_>) -> bool {
        if total <= self.config.target_total_tokens {
            return false;
        }
        if cx.manual {
            return true;
        }
        let ratio_threshold = (cx.budget as f32 * self.config.trigger_context_ratio) as usize;
        total >= ratio_threshold.max(1)
    }

    /// Builds the replacement of a round: its system messages, then one block.
    fn replacement(span: &[BaseMessage], block: BaseMessage) -> Vec<BaseMessage> {
        let mut replacement: Vec<BaseMessage> = span
            .iter()
            .filter(|message| message.role == Role::System)
            .cloned()
            .collect();
        replacement.push(block);
        replacement
    }

    /// Replaces one round with a memory block when that is a real saving.
    async fn compress_span(
        &self,
        messages: &mut Vec<BaseMessage>,
        origin: &mut Vec<usize>,
        cx: &ProcessorContext<'_>,
        span: Range<usize>,
        target_tokens: usize,
        changed: &mut Vec<usize>,
        summaries: &mut Vec<String>,
    ) -> Result<bool> {
        let span_messages = messages[span.clone()].to_vec();
        let before = count_messages(cx.counter, &span_messages);
        let transcript_messages: Vec<BaseMessage> = span_messages
            .iter()
            .filter(|message| message.role != Role::System)
            .cloned()
            .collect();
        if transcript_messages.is_empty() {
            return Ok(false);
        }
        let transcript = summarize_offline(
            &transcript_messages,
            target_tokens.saturating_mul(CHARS_PER_TOKEN_ESTIMATE),
        );
        if cx.cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let model = cx.model.or(self.model.as_ref());
        let max_tokens = u32::try_from(self.config.compression_call_max_tokens).unwrap_or(u32::MAX);
        let summary = summarize(model, COMPRESSION_PROMPT, &transcript, max_tokens, || {
            transcript.clone()
        })
        .await;

        let marker = archive_replaced_messages(cx, &transcript_messages).await;
        let mut block = self.render_block(MemoryBlockKind::RoundLevel.default_scope(), &summary);
        if let Some(marker) = &marker {
            block = marker.append(block);
        }
        let replacement = Self::replacement(&span_messages, BaseMessage::user(block));
        let after = count_messages(cx.counter, &replacement);
        if after >= before {
            tracing::debug!(
                before,
                after,
                "round-level compressor left a round in place: no compression benefit"
            );
            if let Some(marker) = &marker {
                rollback_recall_archive(cx, marker).await;
            }
            return Ok(false);
        }
        let block_origin = origin[span.start];
        let mut new_origins: Vec<usize> = span
            .clone()
            .filter(|index| messages[*index].role == Role::System)
            .map(|index| origin[index])
            .collect();
        new_origins.push(block_origin);
        changed.extend(span.clone().map(|index| origin[index]));
        messages.splice(span.clone(), replacement);
        origin.splice(span, new_origins);
        summaries.push(summary);
        Ok(true)
    }

    /// Builds the shortened history of the hard-truncation phase.
    ///
    /// The prefix keeps its system messages in place, the first compressible
    /// position becomes `block`, and the protected tail follows verbatim.
    fn assemble_truncated(
        messages: &[BaseMessage],
        origin: &[usize],
        compress_end: usize,
        block: &BaseMessage,
    ) -> (Vec<BaseMessage>, Vec<usize>) {
        let mut result: Vec<BaseMessage> = Vec::with_capacity(messages.len());
        let mut result_origin: Vec<usize> = Vec::with_capacity(origin.len());
        let mut inserted = false;
        for index in 0..compress_end {
            if messages[index].role == Role::System {
                result.push(messages[index].clone());
                result_origin.push(origin[index]);
            } else if !inserted {
                result.push(block.clone());
                result_origin.push(origin[index]);
                inserted = true;
            }
        }
        if !inserted {
            return (messages.to_vec(), origin.to_vec());
        }
        result.extend_from_slice(&messages[compress_end..]);
        result_origin.extend_from_slice(&origin[compress_end..]);
        (result, result_origin)
    }

    /// Phase 1: one block per completed round in the compressible prefix.
    async fn phase_one(
        &self,
        messages: &mut Vec<BaseMessage>,
        origin: &mut Vec<usize>,
        cx: &ProcessorContext<'_>,
        changed: &mut Vec<usize>,
        summaries: &mut Vec<String>,
    ) -> Result<()> {
        let compress_end = messages
            .len()
            .saturating_sub(self.config.keep_recent_messages);
        let rounds: Vec<Range<usize>> = group_completed_rounds(messages)
            .into_iter()
            .filter(|round| round.end <= compress_end)
            .collect();
        for span in rounds.into_iter().rev() {
            self.compress_span(
                messages,
                origin,
                cx,
                span,
                self.config.first_pass_target_tokens,
                changed,
                summaries,
            )
            .await?;
        }
        Ok(())
    }

    /// Phase 2: merge contiguous blocks until the history fits or no run can be
    /// merged profitably.
    fn phase_two(
        &self,
        messages: &mut Vec<BaseMessage>,
        origin: &mut Vec<usize>,
        cx: &ProcessorContext<'_>,
        changed: &mut Vec<usize>,
        summaries: &mut Vec<String>,
    ) {
        while count_messages(cx.counter, messages) > self.config.target_total_tokens {
            let runs = iter_summary_merge_ranges(messages, self.marker(), 2);
            if runs.is_empty() {
                break;
            }
            let mut merged_any = false;
            for run in runs.into_iter().rev() {
                let blocks = messages[run.clone()].to_vec();
                let before = count_messages(cx.counter, &blocks);
                // Recall markers of the collapsed blocks survive after the merged
                // body, deduplicated and in their original order.
                let mut markers: Vec<String> = Vec::new();
                let bodies: Vec<String> = blocks
                    .iter()
                    .map(|message| {
                        let (body, found) = split_recall_markers(Self::body(&message.text()));
                        for marker in found {
                            if !markers.contains(&marker) {
                                markers.push(marker);
                            }
                        }
                        body
                    })
                    .collect();
                let merged_body = head_tail_body(
                    &bodies.join(SUMMARY_JOIN_SEPARATOR),
                    self.config
                        .third_pass_target_tokens
                        .saturating_mul(CHARS_PER_TOKEN_ESTIMATE),
                    self.config.truncate_head_ratio,
                    &self.config.truncated_marker,
                );
                let merged = BaseMessage::user(append_recall_markers(
                    self.render_block(MemoryBlockKind::RoundLevel.default_scope(), &merged_body),
                    &markers,
                ));
                if count_message(cx.counter, &merged) >= before {
                    continue;
                }
                let block_origin = origin[run.start];
                changed.extend(run.clone().map(|index| origin[index]));
                messages.splice(run.clone(), std::iter::once(merged));
                origin.splice(run, std::iter::once(block_origin));
                summaries.push(merged_body);
                merged_any = true;
            }
            if !merged_any {
                break;
            }
        }
    }

    /// Phase 3: compress the completed rounds the keep window protected,
    /// excluding the newest completed round.
    ///
    /// The raw rounds are looked up in the tail after the last block, so a
    /// block is never summarised a second time.
    async fn phase_three(
        &self,
        messages: &mut Vec<BaseMessage>,
        origin: &mut Vec<usize>,
        cx: &ProcessorContext<'_>,
        changed: &mut Vec<usize>,
        summaries: &mut Vec<String>,
    ) -> Result<()> {
        if count_messages(cx.counter, messages) <= self.config.target_total_tokens {
            return Ok(());
        }
        let tail_start = messages
            .iter()
            .rposition(|message| self.is_block(message))
            .map_or(0, |index| index + 1);
        let tail = &messages[tail_start..];
        let rounds = group_completed_rounds(tail);
        if rounds.len() <= 1 {
            return Ok(());
        }
        let aggressive: Vec<Range<usize>> = rounds[..rounds.len() - 1]
            .iter()
            .filter(|round| {
                !(*round)
                    .clone()
                    .any(|index| tail[index].role == Role::System)
            })
            .map(|round| tail_start + round.start..tail_start + round.end)
            .collect();
        for span in aggressive.into_iter().rev() {
            self.compress_span(
                messages,
                origin,
                cx,
                span,
                self.config.second_pass_target_tokens,
                changed,
                summaries,
            )
            .await?;
        }
        Ok(())
    }

    /// Phase 4: collapse the compressible prefix into one truncated block.
    ///
    /// Returns the report entry when the replacement was applied. The character
    /// budget starts at the configured token budget minus the surviving system
    /// messages and the protected tail, and is refined against the real counter
    /// so the result lands at or under
    /// [`RoundLevelCompressorConfig::target_total_tokens`] whenever the block
    /// envelope alone fits. The removed prefix is archived for recall, and its
    /// marker is part of the truncated block.
    async fn hard_truncate(
        &self,
        messages: &mut Vec<BaseMessage>,
        origin: &mut Vec<usize>,
        cx: &ProcessorContext<'_>,
        changed: &mut Vec<usize>,
    ) -> Option<String> {
        let compress_end = messages
            .len()
            .saturating_sub(self.config.keep_recent_messages);
        let removable: Vec<usize> = (0..compress_end)
            .filter(|index| messages[*index].role != Role::System)
            .collect();
        if removable.is_empty() {
            return None;
        }
        let transcript_messages: Vec<BaseMessage> = removable
            .iter()
            .map(|index| messages[*index].clone())
            .collect();
        let transcript = summarize_offline(&transcript_messages, usize::MAX);

        let system_tokens = count_messages(
            cx.counter,
            &messages[..compress_end]
                .iter()
                .filter(|message| message.role == Role::System)
                .cloned()
                .collect::<Vec<_>>(),
        );
        let tail_tokens = count_messages(cx.counter, &messages[compress_end..]);
        let budget_tokens = self
            .config
            .target_total_tokens
            .saturating_sub(system_tokens.saturating_add(tail_tokens));
        let ratio = chars_per_token(cx.counter);
        let mut allowed = budget_tokens.saturating_mul(CHARS_PER_TOKEN_ESTIMATE);
        let current_total = count_messages(cx.counter, messages);
        let marker = archive_replaced_messages(cx, &transcript_messages).await;

        let mut candidate: Option<(String, Vec<BaseMessage>, Vec<usize>)> = None;
        for _ in 0..MAX_TRUNCATION_ATTEMPTS {
            let mut body = head_tail_body(
                &transcript,
                allowed,
                self.config.truncate_head_ratio,
                &self.config.truncated_marker,
            );
            if let Some(marker) = &marker {
                body = marker.append(body);
            }
            let block = BaseMessage::user(self.render_block(TRUNCATED_FULL_CONTEXT_SCOPE, &body));
            let (next, next_origin) =
                Self::assemble_truncated(messages, origin, compress_end, &block);
            let total = count_messages(cx.counter, &next);
            let fits = total <= self.config.target_total_tokens || allowed == 0;
            candidate = Some((body, next, next_origin));
            if fits {
                break;
            }
            let overshoot = total - self.config.target_total_tokens;
            let step = overshoot.saturating_mul(ratio);
            if step == 0 {
                break;
            }
            allowed = allowed.saturating_sub(step);
        }

        let Some((body, next, next_origin)) = candidate else {
            if let Some(marker) = &marker {
                rollback_recall_archive(cx, marker).await;
            }
            return None;
        };
        if count_messages(cx.counter, &next) >= current_total {
            tracing::debug!(
                current_total,
                "round-level compressor left the prefix in place: no compression benefit"
            );
            if let Some(marker) = &marker {
                rollback_recall_archive(cx, marker).await;
            }
            return None;
        }
        changed.extend(removable.iter().map(|index| origin[*index]));
        *messages = next;
        *origin = next_origin;
        Some(body)
    }
}

#[async_trait]
impl ContextProcessor for RoundLevelCompressor {
    fn name(&self) -> &'static str {
        "round_level"
    }

    async fn process(
        &self,
        messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        let tokens_before = count_messages(cx.counter, messages);
        if !self.should_run(tokens_before, cx) {
            return Ok(None);
        }
        let original_len = messages.len();
        let mut origin: Vec<usize> = (0..original_len).collect();
        let mut changed: Vec<usize> = Vec::new();
        let mut summaries: Vec<String> = Vec::new();

        self.phase_one(messages, &mut origin, cx, &mut changed, &mut summaries)
            .await?;
        self.phase_two(messages, &mut origin, cx, &mut changed, &mut summaries);
        self.phase_three(messages, &mut origin, cx, &mut changed, &mut summaries)
            .await?;

        // Phase 4: hard truncation of whatever still does not fit.
        if count_messages(cx.counter, messages) > self.config.target_total_tokens {
            if cx.cancellation.is_cancelled() {
                return Err(Error::Cancelled);
            }
            if let Some(body) = self
                .hard_truncate(messages, &mut origin, cx, &mut changed)
                .await
            {
                summaries.push(body);
            }
        }

        if changed.is_empty() {
            return Ok(None);
        }

        changed.sort_unstable();
        changed.dedup();
        let tokens_after = count_messages(cx.counter, messages);
        Ok(Some(ContextEvent {
            processor: self.name().to_owned(),
            kind: ContextEventKind::Summarized,
            messages_changed: changed,
            summary: Some(summaries.join("\n\n")),
            offload_handle: None,
            tokens_before,
            tokens_after,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid_and_round_trips() {
        let config = RoundLevelCompressorConfig::default();
        config.validate().unwrap();
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(
            serde_json::from_str::<RoundLevelCompressorConfig>(&json).unwrap(),
            config
        );
        assert_eq!(
            serde_json::from_str::<RoundLevelCompressorConfig>("{}").unwrap(),
            config
        );
    }

    #[test]
    fn validation_rejects_out_of_range_ratios_and_zero_sizes() {
        let zero_target = RoundLevelCompressorConfig {
            target_total_tokens: 0,
            ..RoundLevelCompressorConfig::default()
        };
        assert!(zero_target.validate().is_err());

        let ratio = RoundLevelCompressorConfig {
            trigger_context_ratio: 0.0,
            ..RoundLevelCompressorConfig::default()
        };
        assert!(ratio.validate().is_err());

        let head = RoundLevelCompressorConfig {
            truncate_head_ratio: 1.0,
            ..RoundLevelCompressorConfig::default()
        };
        assert!(head.validate().is_err());
    }

    #[test]
    fn head_tail_body_keeps_the_marker_and_the_budget() {
        let transcript = "abcdefghijklmnopqrstuvwxyz";
        let body = head_tail_body(transcript, 20, 0.2, "...[X]...");
        assert!(body.contains("...[X]..."));
        assert!(body.chars().count() <= 20);
        assert!(body.starts_with('a'));
        assert!(body.ends_with('z'));
        assert_eq!(head_tail_body(transcript, 0, 0.2, "M"), "M");
        assert_eq!(head_tail_body("short", 100, 0.2, "M"), "short");
    }

    #[test]
    fn the_character_ratio_follows_the_counter() {
        use openjiuwen_foundation::prompt::ApproxTokenCounter;
        assert_eq!(
            chars_per_token(&ApproxTokenCounter::with_chars_per_token(1)),
            1
        );
        assert_eq!(
            chars_per_token(&ApproxTokenCounter::with_chars_per_token(4)),
            4
        );
    }
}
