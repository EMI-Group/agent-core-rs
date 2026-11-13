//! Token and cost accounting behind the `/cost` and `/usage` commands.
//!
//! [`UsageTotals`] folds the frames a `DeepAgent` turn streams into per-session
//! counters and renders them as plain text (no ANSI, no money: the CLI has no
//! price table, so `/cost` reports tokens and call counts only).
//!
//! The two frame kinds that carry accounting data are:
//!
//! - `llm_usage` (`openjiuwen_agent::output::LLM_USAGE`), whose payload holds
//!   `usage_metadata` with `input_tokens`, `output_tokens`, `total_tokens`,
//!   `cache_tokens` and `reasoning_tokens`, or those counters at the top level of
//!   the payload when the provider omitted the wrapper;
//! - `answer` (`openjiuwen_agent::output::ANSWER`), whose payload `observations`
//!   array is the only place the kernel reports tool calls on the harness stream
//!   — the `tool_call`/`tool_result` frame kinds the renderer knows about never
//!   appear on `DeepAgent::stream`, so counting them would always yield zero.
//!
//! Turns are not a frame: the REPL calls [`UsageTotals::record_turn`] once per
//! completed turn. Every other frame kind is ignored, so new kernel frames
//! degrade to silence instead of noise.

use openjiuwen_agent::output::{AgentChunk, ANSWER, LLM_USAGE};
use serde_json::Value;

/// Per-session token and call accounting for `/cost` and `/status`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UsageTotals {
    /// Prompt tokens consumed by every model call.
    pub input_tokens: u64,
    /// Completion tokens produced by every model call.
    pub output_tokens: u64,
    /// Total tokens as reported per call, or the input/output sum when absent.
    pub total_tokens: u64,
    /// Prompt tokens served from the provider's prompt cache.
    pub cache_tokens: u64,
    /// Output tokens spent on reasoning/thinking.
    pub reasoning_tokens: u64,
    /// Number of `llm_usage` frames folded in, i.e. model calls.
    pub model_calls: u64,
    /// Number of tool calls reported by `answer` frames.
    pub tool_calls: u64,
    /// Number of completed turns, counted by [`UsageTotals::record_turn`].
    pub turns: u64,
}

impl UsageTotals {
    /// Creates zeroed totals.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Folds one streamed frame into the totals.
    ///
    /// `llm_usage` frames add their counters and one model call; `answer` frames
    /// add the length of their `observations` array to the tool-call count; every
    /// other kind is ignored. Counters are read as unsigned integers and missing
    /// ones count as zero; a missing `total_tokens` falls back to
    /// `input_tokens + output_tokens` for that frame. Additions saturate, so a
    /// pathological stream can never wrap a counter.
    pub fn record(&mut self, chunk: &AgentChunk) {
        match chunk.kind.as_str() {
            LLM_USAGE => self.record_usage(&chunk.payload),
            ANSWER => {
                let calls = chunk
                    .payload
                    .get("observations")
                    .and_then(Value::as_array)
                    .map_or(0, Vec::len);
                self.tool_calls = self
                    .tool_calls
                    .saturating_add(u64::try_from(calls).unwrap_or(0));
            }
            _ => {}
        }
    }

    /// Counts one completed turn.
    pub fn record_turn(&mut self) {
        self.turns = self.turns.saturating_add(1);
    }

    /// Adds `other` into these totals, field by field.
    pub fn merge(&mut self, other: &Self) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.total_tokens = self.total_tokens.saturating_add(other.total_tokens);
        self.cache_tokens = self.cache_tokens.saturating_add(other.cache_tokens);
        self.reasoning_tokens = self.reasoning_tokens.saturating_add(other.reasoning_tokens);
        self.model_calls = self.model_calls.saturating_add(other.model_calls);
        self.tool_calls = self.tool_calls.saturating_add(other.tool_calls);
        self.turns = self.turns.saturating_add(other.turns);
    }

    /// Resets every counter to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Returns whether every counter is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.input_tokens == 0
            && self.output_tokens == 0
            && self.total_tokens == 0
            && self.cache_tokens == 0
            && self.reasoning_tokens == 0
            && self.model_calls == 0
            && self.tool_calls == 0
            && self.turns == 0
    }

    /// Renders the plain-text `/cost` block.
    ///
    /// The block is exactly:
    ///
    /// ```text
    /// Token usage:
    ///   Input:  1,234 tokens
    ///   Output: 567 tokens
    ///   Total:  1,801 tokens
    ///   Calls:  3
    ///   Tool calls: 2
    /// ```
    ///
    /// Counters are grouped with `,` thousands separators. Cache and reasoning
    /// tokens are accumulated but not printed, because the Python CLI does not
    /// show them either.
    #[must_use]
    pub fn render(&self) -> String {
        format!(
            "Token usage:\n  Input:  {input} tokens\n  Output: {output} tokens\n  Total:  {total} tokens\n  Calls:  {calls}\n  Tool calls: {tools}",
            input = grouped(self.input_tokens),
            output = grouped(self.output_tokens),
            total = grouped(self.total_tokens),
            calls = self.model_calls,
            tools = self.tool_calls,
        )
    }

    /// Renders the one-line summary used by `/status`.
    #[must_use]
    pub fn render_line(&self) -> String {
        format!(
            "Input {input} | Output {output} | Total {total} tokens | Calls {calls} | Tool calls {tools}",
            input = grouped(self.input_tokens),
            output = grouped(self.output_tokens),
            total = grouped(self.total_tokens),
            calls = self.model_calls,
            tools = self.tool_calls,
        )
    }

    /// Folds one `llm_usage` payload into the totals.
    fn record_usage(&mut self, payload: &Value) {
        let source = payload.get("usage_metadata").unwrap_or(payload);
        let input_tokens = counter(source, "input_tokens");
        let output_tokens = counter(source, "output_tokens");
        let total_tokens = source
            .get("total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(input_tokens.saturating_add(output_tokens));

        self.input_tokens = self.input_tokens.saturating_add(input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(output_tokens);
        self.total_tokens = self.total_tokens.saturating_add(total_tokens);
        self.cache_tokens = self
            .cache_tokens
            .saturating_add(counter(source, "cache_tokens"));
        self.reasoning_tokens = self
            .reasoning_tokens
            .saturating_add(counter(source, "reasoning_tokens"));
        self.model_calls = self.model_calls.saturating_add(1);
    }
}

/// Returns the unsigned integer stored under `key`, or zero when it is absent or
/// not an unsigned integer.
fn counter(source: &Value, key: &str) -> u64 {
    source.get(key).and_then(Value::as_u64).unwrap_or(0)
}

/// Formats `value` with `,` thousands separators: `1234567` → `1,234,567`.
fn grouped(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.char_indices() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use serde_json::json;

    use super::*;

    /// Builds a frame of `kind` carrying `payload`.
    fn frame(kind: &str, payload: Value) -> AgentChunk {
        AgentChunk {
            kind: kind.to_owned(),
            index: 0,
            payload,
        }
    }

    #[test]
    fn accumulates_usage_metadata_and_bare_payload_frames() {
        let mut totals = UsageTotals::new();
        totals.record(&frame(
            LLM_USAGE,
            json!({"usage_metadata": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "cache_tokens": 7,
                "reasoning_tokens": 3,
            }}),
        ));
        totals.record(&frame(
            LLM_USAGE,
            json!({"input_tokens": 10, "output_tokens": 5}),
        ));

        assert_eq!(
            totals,
            UsageTotals {
                input_tokens: 110,
                output_tokens: 25,
                total_tokens: 135,
                cache_tokens: 7,
                reasoning_tokens: 3,
                model_calls: 2,
                tool_calls: 0,
                turns: 0,
            }
        );
        assert!(!totals.is_empty());
    }

    #[test]
    fn total_tokens_falls_back_to_the_input_output_sum() {
        let mut totals = UsageTotals::new();
        totals.record(&frame(
            LLM_USAGE,
            json!({"usage_metadata": {"input_tokens": 4, "output_tokens": 6}}),
        ));
        assert_eq!(totals.total_tokens, 10);
        assert_eq!(totals.model_calls, 1);
    }

    #[test]
    fn answer_observations_count_tool_calls() {
        let mut totals = UsageTotals::new();
        totals.record(&frame(
            ANSWER,
            json!({"observations": [{"tool": "bash"}, {"tool": "read_file"}]}),
        ));
        // An answer without observations adds nothing and never panics.
        totals.record(&frame(ANSWER, json!({"observations": "not an array"})));
        assert_eq!(totals.tool_calls, 2);
        assert_eq!(totals.model_calls, 0);
    }

    #[test]
    fn unknown_kinds_and_malformed_payloads_are_ignored() {
        let mut totals = UsageTotals::new();
        for (kind, payload) in [
            ("llm_output", json!({"content": "hi"})),
            (
                LLM_USAGE,
                json!({"usage_metadata": {"input_tokens": "many"}}),
            ),
            ("tool_call", json!({"name": "bash"})),
        ] {
            totals.record(&frame(kind, payload));
        }
        assert_eq!(totals.input_tokens, 0);
        assert_eq!(totals.output_tokens, 0);
        assert_eq!(totals.total_tokens, 0);
        // The malformed usage frame still counts as a model call.
        assert_eq!(totals.model_calls, 1);
        assert_eq!(totals.tool_calls, 0);
    }

    #[test]
    fn record_turn_merge_and_reset() {
        let mut totals = UsageTotals::new();
        totals.record_turn();
        totals.record_turn();
        totals.record(&frame(LLM_USAGE, json!({"total_tokens": 1})));
        assert_eq!(totals.turns, 2);

        let other = UsageTotals {
            input_tokens: 3,
            output_tokens: 4,
            total_tokens: 7,
            cache_tokens: 1,
            reasoning_tokens: 2,
            model_calls: 5,
            tool_calls: 6,
            turns: 7,
        };
        totals.merge(&other);
        assert_eq!(totals.turns, 9);
        assert_eq!(totals.model_calls, 6);
        assert_eq!(totals.tool_calls, 6);
        assert_eq!(totals.total_tokens, 8);
        assert_eq!(totals.reasoning_tokens, 2);
        assert!(!totals.is_empty());

        totals.reset();
        assert!(totals.is_empty());
        assert_eq!(totals, UsageTotals::default());
    }

    #[test]
    fn thousands_separators_group_every_three_digits() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(7), "7");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1000), "1,000");
        assert_eq!(grouped(1_234_567), "1,234,567");
        assert_eq!(grouped(u64::MAX), "18,446,744,073,709,551,615");
    }

    #[test]
    fn render_and_render_line_use_the_documented_shapes() {
        let totals = UsageTotals {
            input_tokens: 1_234,
            output_tokens: 567,
            total_tokens: 1_801,
            model_calls: 3,
            tool_calls: 2,
            ..UsageTotals::default()
        };
        assert_eq!(
            totals.render(),
            "Token usage:\n  Input:  1,234 tokens\n  Output: 567 tokens\n  Total:  1,801 tokens\n  Calls:  3\n  Tool calls: 2"
        );
        assert_eq!(
            totals.render_line(),
            "Input 1,234 | Output 567 | Total 1,801 tokens | Calls 3 | Tool calls 2"
        );
        assert_eq!(
            UsageTotals::new().render_line(),
            "Input 0 | Output 0 | Total 0 tokens | Calls 0 | Tool calls 0"
        );
    }
}
