//! Chunk-to-terminal renderer shared by the one-shot runner and the REPL.
//!
//! [`RenderState`] accumulates one streamed turn: it keeps the model output, the
//! final answer, token usage and any interrupt requests, and writes the frames a
//! human should see. Passing `out: None` makes it collect-only, which is how the
//! JSON output formats consume the stream without printing anything.
//!
//! Frame handling follows the kernel frame kinds ([`LLM_OUTPUT`],
//! [`LLM_REASONING`], [`LLM_USAGE`], [`ANSWER`], [`INTERACTION`]) plus the
//! tool-call frames the agent emits. Anything else is recorded in
//! [`RenderState::chunk_types`] and never printed, so new frame kinds degrade to
//! silence instead of noise.

use openjiuwen_agent::output::{
    AgentChunk, ANSWER, INTERACTION, LLM_OUTPUT, LLM_REASONING, LLM_USAGE,
};
use openjiuwen_common::error::Result;
use serde_json::{json, Value};

use super::io::CliIo;

/// Frame kind of a tool invocation.
const TOOL_CALL: &str = "tool_call";
/// Frame kind of a tool result.
const TOOL_RESULT: &str = "tool_result";
/// Longest tool-result summary printed before it is elided.
const SUMMARY_LIMIT: usize = 2000;

/// Accumulates one streamed turn and renders it to the terminal.
#[derive(Debug, Clone, Default)]
pub struct RenderState {
    /// Accumulated `llm_output` text.
    pub text: String,
    /// Kinds of frames that had no dedicated handling.
    pub chunk_types: Vec<String>,
    /// Last captured `llm_usage` payload.
    pub usage: Option<Value>,
    /// Recorded `__interaction__` frames as `{"id": ..., "value": ...}`.
    pub interactions: Vec<Value>,
    /// Result type reported by the `answer` frame.
    pub result_type: Option<String>,
    /// Accumulated `answer` frame text, shown only when nothing streamed.
    answer: String,
}

impl RenderState {
    /// Creates an empty state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies one streamed frame, writing it to `out` when a terminal is given.
    ///
    /// # Errors
    ///
    /// Returns the writer error when a rendered frame cannot be written.
    pub async fn render(
        &mut self,
        chunk: &AgentChunk,
        out: Option<&CliIo>,
        verbose: bool,
    ) -> Result<()> {
        match chunk.kind.as_str() {
            LLM_OUTPUT => self.render_output(chunk, out).await,
            ANSWER => self.render_answer(chunk, out).await,
            LLM_REASONING => self.render_reasoning(chunk, out, verbose).await,
            LLM_USAGE => {
                self.capture_usage(chunk);
                Ok(())
            }
            TOOL_CALL => self.render_tool_call(chunk, out).await,
            TOOL_RESULT => self.render_tool_result(chunk, out).await,
            INTERACTION => {
                self.record_interaction(chunk);
                Ok(())
            }
            _ => {
                self.chunk_types.push(chunk.kind.clone());
                Ok(())
            }
        }
    }

    /// Returns the answer of the turn.
    ///
    /// This is the streamed model output when there was any, otherwise the
    /// `answer` frame's output.
    #[must_use]
    pub fn answer(&self) -> &str {
        if self.text.is_empty() {
            &self.answer
        } else {
            &self.text
        }
    }

    /// Returns the result type of the turn, defaulting to `answer`.
    #[must_use]
    pub fn result_type(&self) -> &str {
        self.result_type.as_deref().unwrap_or("answer")
    }

    /// Terminates the rendered turn.
    ///
    /// Ensures the streamed text ends with a newline; nothing is written when no
    /// text was streamed or the text already ends with one.
    ///
    /// # Errors
    ///
    /// Returns the writer error when the newline cannot be written.
    pub async fn finish(&mut self, out: Option<&CliIo>) -> Result<()> {
        let Some(out) = out else {
            return Ok(());
        };
        if self.text.is_empty() || self.text.ends_with('\n') {
            return Ok(());
        }
        out.write("\n").await
    }

    /// Streams `llm_output` text verbatim.
    async fn render_output(&mut self, chunk: &AgentChunk, out: Option<&CliIo>) -> Result<()> {
        let Some(content) = chunk.payload.get("content").and_then(Value::as_str) else {
            return Ok(());
        };
        self.text.push_str(content);
        match out {
            Some(out) => out.write(content).await,
            None => Ok(()),
        }
    }

    /// Captures the final answer and prints it when nothing was streamed.
    async fn render_answer(&mut self, chunk: &AgentChunk, out: Option<&CliIo>) -> Result<()> {
        let output = chunk.payload.get("output").and_then(Value::as_str);
        if let Some(output) = output {
            output.clone_into(&mut self.answer);
        }
        self.result_type = chunk
            .payload
            .get("result_type")
            .and_then(Value::as_str)
            .map(str::to_owned);
        match out {
            Some(out) if self.text.is_empty() => match output {
                Some(output) if !output.is_empty() => out.write_line(output).await,
                _ => Ok(()),
            },
            _ => Ok(()),
        }
    }

    /// Prints reasoning text only in verbose mode.
    async fn render_reasoning(
        &mut self,
        chunk: &AgentChunk,
        out: Option<&CliIo>,
        verbose: bool,
    ) -> Result<()> {
        let Some(out) = out else {
            return Ok(());
        };
        if !verbose {
            return Ok(());
        }
        let Some(content) = chunk.payload.get("content").and_then(Value::as_str) else {
            return Ok(());
        };
        out.write_line(content).await
    }

    /// Records the usage payload, preferring the kernel's `usage_metadata`.
    fn capture_usage(&mut self, chunk: &AgentChunk) {
        let payload = &chunk.payload;
        let usage = payload
            .get("usage_metadata")
            .or_else(|| payload.get("usage"))
            .cloned()
            .or_else(|| {
                if payload.is_object() {
                    Some(payload.clone())
                } else {
                    None
                }
            });
        if usage.is_some() {
            self.usage = usage;
        }
    }

    /// Prints `● {name}({args})`.
    async fn render_tool_call(&mut self, chunk: &AgentChunk, out: Option<&CliIo>) -> Result<()> {
        let Some(out) = out else {
            return Ok(());
        };
        let payload = &chunk.payload;
        let name = payload
            .get("name")
            .and_then(Value::as_str)
            .or_else(|| payload.get("tool_name").and_then(Value::as_str))
            .or_else(|| {
                payload
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
            })
            .unwrap_or("unknown");
        let arguments = payload
            .get("arguments")
            .or_else(|| payload.get("args"))
            .or_else(|| payload.get("input"))
            .unwrap_or(&Value::Null);
        out.write_line(&format!("● {name}({})", compact(arguments)))
            .await
    }

    /// Prints `  ⎿  {summary}` followed by a blank line.
    async fn render_tool_result(&mut self, chunk: &AgentChunk, out: Option<&CliIo>) -> Result<()> {
        let Some(out) = out else {
            return Ok(());
        };
        let payload = &chunk.payload;
        let summary = ["rendered_result", "result", "output", "summary"]
            .iter()
            .find_map(|key| payload.get(*key))
            .map_or_else(String::new, compact);
        let summary = truncate(&summary);
        out.write(&format!("  ⎿  {summary}\n\n")).await
    }

    /// Records an interaction request; it is answered elsewhere, never printed.
    fn record_interaction(&mut self, chunk: &AgentChunk) {
        self.interactions.push(json!({
            "id": chunk.payload.get("id").cloned().unwrap_or(Value::Null),
            "value": chunk.payload.get("value").cloned().unwrap_or(Value::Null),
        }));
    }
}

/// Renders a JSON value: raw text for strings, compact JSON otherwise.
fn compact(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Shortens `text` to [`SUMMARY_LIMIT`] characters, marking the elision.
fn truncate(text: &str) -> String {
    if text.chars().count() <= SUMMARY_LIMIT {
        return text.to_owned();
    }
    let mut shortened: String = text.chars().take(SUMMARY_LIMIT).collect();
    shortened.push_str("...");
    shortened
}
