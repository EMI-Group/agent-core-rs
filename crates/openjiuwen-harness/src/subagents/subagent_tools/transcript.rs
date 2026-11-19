//! The additive `subagent_transcript` tool: the model-facing read of a sub-agent transcript.
//!
//! Python has no transcript tool; this is a Rust-port extension, which is why it is not part
//! of the six-tool parity set returned by [`crate::subagents::subagent_tools`].
//! [`crate::subagents::subagent_tools_with_transcript`] mounts it alongside those six tools.
//!
//! The read is bounded on both sides: [`SubAgentRuntime::transcript_after`] paginates by
//! sequence number and `limit` is clamped to `1..=200`, so a chatty sub-agent can never flood
//! the parent's context window in one call.

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
use serde_json::{json, Value};

use super::super::runtime::SubAgentRuntime;
use super::super::task_tool::description_for;
use super::super::RuntimeSubAgentSpawner;
use super::util::{non_empty, payload_error};
use crate::tools::{tool_card, ToolPayload};

/// Entries returned when `limit` is absent.
const DEFAULT_TRANSCRIPT_LIMIT: usize = 20;

/// Largest page [`SubagentTranscriptTool`] returns; larger requests are clamped.
const MAX_TRANSCRIPT_LIMIT: usize = 200;

subagent_tool!(
    /// Reads the bounded transcript of one sub-agent instance.
    SubagentTranscriptTool,
    "subagent_transcript",
    json!({
        "type": "object",
        "properties": {
            "subagent_id": {
                "type": "string",
                "description": "Subagent instance id."
            },
            "limit": {
                "type": "integer",
                "default": 20,
                "description": "Maximum entries to return; clamped to 1..=200."
            },
            "after_seq": {
                "type": "integer",
                "default": 0,
                "description": "Return only entries with a larger sequence number."
            }
        },
        "required": ["subagent_id"]
    })
);

#[async_trait]
impl Tool for SubagentTranscriptTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        let Some(subagent_id) = non_empty(&inputs, "subagent_id") else {
            return Ok(ToolPayload::fail("'subagent_id' must be a non-empty string").to_json());
        };
        let limit = match resolve_limit(&inputs) {
            Ok(limit) => limit,
            Err(message) => return Ok(ToolPayload::fail(message).to_json()),
        };
        let after_seq = match resolve_after_seq(&inputs) {
            Ok(after_seq) => after_seq,
            Err(message) => return Ok(ToolPayload::fail(message).to_json()),
        };
        let Some(summary) = self.runtime().transcript_summary(&subagent_id) else {
            return Ok(ToolPayload::fail(format!("subagent not found: {subagent_id}")).to_json());
        };
        let entries = self
            .runtime()
            .transcript_after(&subagent_id, after_seq, limit)
            .unwrap_or_default();
        let count = entries.len();
        let last_seq = entries.last().map_or(after_seq, |entry| entry.seq);
        let has_more = self
            .runtime()
            .transcript_after(&subagent_id, last_seq, 1)
            .is_some_and(|more| !more.is_empty());
        let activity = self
            .runtime()
            .get_metadata(&subagent_id)
            .map_or(Value::Null, |metadata| {
                Value::String(metadata.activity.as_str().to_owned())
            });
        Ok(ToolPayload::ok(json!({
            "subagent_id": subagent_id,
            "status": self.runtime().get_status(&subagent_id).as_str(),
            "activity": activity,
            "entries": entries,
            "count": count,
            "total": summary.count,
            "dropped": summary.dropped,
            "last_kind": summary.last_kind,
            "has_more": has_more,
        }))
        .to_json())
    }

    fn render_for_llm(&self, output: &Value) -> String {
        if let Some(error) = payload_error(output) {
            return error.to_owned();
        }
        let Some(data) = output.get("data") else {
            return String::new();
        };
        let id = data
            .get("subagent_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let status = data
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let total = data
            .get("total")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let dropped = data
            .get("dropped")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let noun = if total == 1 { "entry" } else { "entries" };
        let mut lines = vec![if dropped > 0 {
            format!("Transcript of {id} ({status}, {total} {noun}, {dropped} dropped)")
        } else {
            format!("Transcript of {id} ({status}, {total} {noun})")
        }];
        if let Some(entries) = data.get("entries").and_then(Value::as_array) {
            lines.extend(entries.iter().map(render_entry));
        }
        let mut rendered = lines.join("\n");
        if data.get("has_more").and_then(Value::as_bool) == Some(true) {
            let count = data
                .get("count")
                .and_then(Value::as_u64)
                .unwrap_or_default();
            let remaining = total.saturating_sub(count);
            rendered = format!("{rendered}\n… {remaining} more entries");
        }
        rendered
    }
}

/// Resolves the page size: absent keeps the default, a non-integer or negative value is an
/// error, and any other value is clamped to `1..=MAX_TRANSCRIPT_LIMIT`.
fn resolve_limit(inputs: &Value) -> std::result::Result<usize, &'static str> {
    match inputs.get("limit") {
        None => Ok(DEFAULT_TRANSCRIPT_LIMIT),
        Some(value) => value
            .as_u64()
            .map(|limit| {
                usize::try_from(limit)
                    .unwrap_or(MAX_TRANSCRIPT_LIMIT)
                    .clamp(1, MAX_TRANSCRIPT_LIMIT)
            })
            .ok_or("'limit' must be an integer"),
    }
}

/// Resolves the pagination cursor; absent means "from the oldest retained entry".
fn resolve_after_seq(inputs: &Value) -> std::result::Result<u64, &'static str> {
    match inputs.get("after_seq") {
        None => Ok(0),
        Some(value) => value
            .as_u64()
            .ok_or("'after_seq' must be a non-negative integer"),
    }
}

/// Renders one transcript entry: `[<seq>] <kind> <role>: <content>` plus the tool outcome.
fn render_entry(entry: &Value) -> String {
    let seq = entry.get("seq").and_then(Value::as_u64).unwrap_or_default();
    let kind = entry
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or("status");
    let role = entry
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let content = entry
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut line = format!("[{seq}] {kind} {role}: {content}");
    if let Some(tool_name) = entry.get("tool_name").and_then(Value::as_str) {
        if let Some(success) = entry.get("success").and_then(Value::as_bool) {
            let outcome = if success { "ok" } else { "failed" };
            line = format!("{line} ({tool_name}, {outcome})");
        }
    }
    line
}
