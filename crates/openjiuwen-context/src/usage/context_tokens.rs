//! Shared context-window sizing policy.
//!
//! Ports `usage/context_tokens.py`. These helpers are the only part of the
//! usage package that compression and capacity checks share, so they keep
//! Python's three-rung policy:
//!
//! 1. a last model-reported usage baseline plus the estimated appended tail,
//! 2. the supplied [`TokenCounter`],
//! 3. `len(text) // 3` per message.
//!
//! The baseline rung is usage-aware on purpose: a provider total already
//! includes the system prompt and tools of that call, so a partial message span
//! must never use it. Callers therefore pass `usage_aware = true` only for a
//! full cumulative list.

use openjiuwen_foundation::prompt::{BaseMessage, MessageContent, Role, TokenCounter};
use serde_json::Value;

/// The message-metadata key holding a provider usage record.
pub const USAGE_METADATA_KEY: &str = "usage_metadata";

/// The message-metadata flag marking a usage record as no longer describing its
/// prefix.
pub const USAGE_STALE_METADATA_KEY: &str = "context_usage_stale";

/// Characters per token of the last-resort estimate.
pub const CHARACTERS_PER_ESTIMATED_TOKEN: usize = 3;

/// Estimates tokens of sanitized text as `len // 3`, with a floor of one token.
///
/// Counting is by Unicode scalar value, matching Python's `len(text)`, so
/// multi-byte text is not over-counted.
#[must_use]
pub fn estimate_text_tokens(text: &str) -> u64 {
    let estimate = text.chars().count() / CHARACTERS_PER_ESTIMATED_TOKEN;
    u64::try_from(estimate.max(1)).unwrap_or(u64::MAX)
}

/// Estimates tokens of a content payload without a counter.
///
/// The payload is flattened to text first ([`MessageContent::to_plain_text`],
/// the same view [`BaseMessage::text`] returns), so a base64 image payload is
/// never counted as prompt text.
#[must_use]
pub fn estimate_content_tokens(content: &MessageContent) -> u64 {
    estimate_text_tokens(&content.to_plain_text())
}

/// Estimates tokens of one message without a counter.
#[must_use]
pub fn estimate_message_tokens(message: &BaseMessage) -> u64 {
    estimate_text_tokens(&message.text())
}

/// The provider-reported total tokens of a message, when it carries one.
///
/// The record lives under [`USAGE_METADATA_KEY`] as an object with a
/// `total_tokens` counter; numbers and digit strings are accepted.
#[must_use]
pub fn usage_total_tokens(message: &BaseMessage) -> Option<u64> {
    let usage = message.metadata.get(USAGE_METADATA_KEY)?;
    read_u64(usage.get("total_tokens")?)
}

/// Whether a message's usage record still describes its prompt prefix.
///
/// Mirrors Python's `ContextUtils.has_valid_usage_metadata`: the message must be
/// an assistant message with a positive reported total, and its metadata must
/// not carry the stale flag.
#[must_use]
pub fn has_valid_usage_metadata(message: &BaseMessage) -> bool {
    if message.role != Role::Assistant {
        return false;
    }
    if message
        .metadata
        .get(USAGE_STALE_METADATA_KEY)
        .is_some_and(json_truthy)
    {
        return false;
    }
    usage_total_tokens(message).is_some_and(|tokens| tokens > 0)
}

/// Returns the last valid model-reported total plus the estimated appended tail.
///
/// The tail is estimated at `len(text) // 3` per message with no floor, exactly
/// like Python: a message that contributes no text at all contributes no tokens.
#[must_use]
pub fn count_usage_tokens_with_tail(messages: &[BaseMessage]) -> Option<u64> {
    for (index, message) in messages.iter().enumerate().rev() {
        if !has_valid_usage_metadata(message) {
            continue;
        }
        let usage_tokens = usage_total_tokens(message)?;
        let tail_tokens = messages[index.saturating_add(1)..]
            .iter()
            .map(|item| {
                u64::try_from(item.text().chars().count() / CHARACTERS_PER_ESTIMATED_TOKEN)
                    .unwrap_or(u64::MAX)
            })
            .fold(0u64, u64::saturating_add);
        return Some(usage_tokens.saturating_add(tail_tokens));
    }
    None
}

/// Counts messages with the context tokenizer and its character fallback.
///
/// `usage_aware` must only be `true` for a full cumulative message list: a
/// model-reported usage total cannot size an isolated slice.
#[must_use]
pub fn count_messages_tokens(
    messages: &[BaseMessage],
    counter: Option<&dyn TokenCounter>,
    usage_aware: bool,
) -> u64 {
    if messages.is_empty() {
        return 0;
    }
    if usage_aware {
        if let Some(usage_tokens) = count_usage_tokens_with_tail(messages) {
            return usage_tokens;
        }
    }
    if let Some(counter) = counter {
        return u64::try_from(counter.count_messages(messages)).unwrap_or(u64::MAX);
    }
    messages
        .iter()
        .map(estimate_message_tokens)
        .fold(0u64, u64::saturating_add)
}

/// Counts tool definitions with the context tokenizer or its estimate.
///
/// The counted payload is the canonical JSON of each definition, matching the
/// sort-keys serialization Python measures.
#[must_use]
pub fn count_tools_tokens(tools: &[Value], counter: Option<&dyn TokenCounter>) -> u64 {
    tools
        .iter()
        .map(|tool| {
            let text = canonical_json(tool);
            match counter {
                Some(counter) => u64::try_from(counter.count_text(&text)).unwrap_or(u64::MAX),
                None => estimate_text_tokens(&text),
            }
        })
        .fold(0u64, u64::saturating_add)
}

/// The compressor's usage-baseline, tokenizer, and estimate policy.
///
/// A model-reported baseline already includes the system prompt and tools of
/// its call, so it wins outright; otherwise the messages and the tool
/// definitions are counted separately and summed.
#[must_use]
pub fn count_context_window_tokens(
    messages: &[BaseMessage],
    tools: &[Value],
    counter: Option<&dyn TokenCounter>,
) -> u64 {
    if let Some(usage_tokens) = count_usage_tokens_with_tail(messages) {
        return usage_tokens;
    }
    count_messages_tokens(messages, counter, false)
        .saturating_add(count_tools_tokens(tools, counter))
}

/// Renders a JSON value with recursively sorted object keys.
///
/// `serde_json` is built with `preserve_order`, so insertion order would leak
/// into cache keys and tool estimates; sorting makes both deterministic.
#[must_use]
pub fn canonical_json(value: &Value) -> String {
    let mut serialized = String::new();
    write_canonical(value, &mut serialized);
    serialized
}

/// Appends the canonical rendering of `value` to `out`.
fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            out.push('{');
            for (index, (key, item)) in entries.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(key.clone()).to_string());
                out.push(':');
                write_canonical(item, out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}

/// Reads a non-negative integer from JSON numbers and digit strings.
fn read_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number
            .as_u64()
            .or_else(|| number.as_i64().and_then(|inner| u64::try_from(inner).ok()))
            .or_else(|| {
                let float = number.as_f64()?;
                let integral = (float - float.round()).abs() < f64::EPSILON;
                if integral && float.is_finite() && float >= 0.0 {
                    Some(float as u64)
                } else {
                    None
                }
            }),
        Value::String(text) => {
            let trimmed = text.trim();
            let digits = trimmed.strip_prefix('+').unwrap_or(trimmed);
            if digits.is_empty() || !digits.chars().all(|character| character.is_ascii_digit()) {
                return None;
            }
            digits.parse::<u64>().ok()
        }
        _ => None,
    }
}

/// Python-truthiness of a JSON value, used for boolean provider flags.
pub(crate) fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|float| float != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}
