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
//!
//! Styling is opt-in through [`RenderState::styled`]: the chrome (tool call, tool
//! result, verbose reasoning) is coloured, while the answer text itself never is,
//! so a run's output can still be piped. A plain [`RenderState`] emits exactly
//! the bytes it emitted before styling existed.

use openjiuwen_agent::output::{
    AgentChunk, ANSWER, INTERACTION, LLM_OUTPUT, LLM_REASONING, LLM_USAGE,
};
use openjiuwen_common::error::Result;
use serde_json::{json, Value};

use super::io::CliIo;
use super::style::Style;

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
    /// Styling applied to rendered chrome; [`Style::plain`] by default.
    style: Style,
}

impl RenderState {
    /// Creates an empty, unstyled state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty state whose chrome is styled when `enabled` is true.
    #[must_use]
    pub fn styled(enabled: bool) -> Self {
        Self {
            style: Style::new(enabled),
            ..Self::default()
        }
    }

    /// Returns the styling this state renders with.
    #[must_use]
    pub fn style(&self) -> Style {
        self.style
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

    /// Streams `llm_output` text verbatim, never coloured.
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
    ///
    /// The printed answer is never coloured: it is model text meant for piping.
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

    /// Prints reasoning text, dimmed, only in verbose mode.
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
        let content = self.style.dim(content);
        out.write_line(&content).await
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

    /// Prints `● {name}` in cyan followed by `({args})` in dim.
    ///
    /// Without styling the line is exactly `● {name}({args})`.
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
        let name = self.style.cyan(&format!("● {name}"));
        let arguments = self.style.dim(&format!("({})", compact(arguments)));
        out.write_line(&format!("{name}{arguments}")).await
    }

    /// Prints `  ⎿  {summary}`, dimmed, followed by a blank line.
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
        let summary = self.style.dim(&summary);
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::super::io::test_support::{output, scripted_io};
    use super::*;

    /// Renders `frame` through a fresh state and returns everything written.
    async fn rendered_frame(state: &mut RenderState, frame: &AgentChunk, verbose: bool) -> String {
        let (io, buffer) = scripted_io(&[]);
        state.render(frame, Some(&io), verbose).await.unwrap();
        output(&buffer)
    }

    /// Builds a `tool_call` frame for `bash`.
    fn tool_call_frame() -> AgentChunk {
        AgentChunk::new(
            TOOL_CALL,
            0,
            json!({"name": "bash", "arguments": {"command": "ls"}}),
        )
    }

    #[tokio::test]
    async fn plain_frames_render_byte_exactly() {
        let mut state = RenderState::new();
        assert_eq!(state.style(), Style::plain());
        assert_eq!(
            rendered_frame(&mut state, &tool_call_frame(), false).await,
            "● bash({\"command\":\"ls\"})\n"
        );

        let mut state = RenderState::new();
        let result = AgentChunk::new(TOOL_RESULT, 1, json!({"result": "ok"}));
        assert_eq!(
            rendered_frame(&mut state, &result, false).await,
            "  ⎿  ok\n\n"
        );

        let mut state = RenderState::new();
        let reasoning = AgentChunk::new(LLM_REASONING, 0, json!({"content": "thinking"}));
        assert_eq!(
            rendered_frame(&mut state, &reasoning, true).await,
            "thinking\n"
        );
    }

    #[tokio::test]
    async fn styled_frames_carry_ansi_prefixes() {
        let mut state = RenderState::styled(true);
        assert!(state.style().enabled());
        assert_eq!(
            rendered_frame(&mut state, &tool_call_frame(), false).await,
            "\x1b[36m● bash\x1b[0m\x1b[2m({\"command\":\"ls\"})\x1b[0m\n"
        );

        let mut state = RenderState::styled(true);
        let result = AgentChunk::new(TOOL_RESULT, 1, json!({"result": "ok"}));
        assert_eq!(
            rendered_frame(&mut state, &result, false).await,
            "  ⎿  \x1b[2mok\x1b[0m\n\n"
        );

        let mut state = RenderState::styled(true);
        let reasoning = AgentChunk::new(LLM_REASONING, 0, json!({"content": "thinking"}));
        assert_eq!(
            rendered_frame(&mut state, &reasoning, true).await,
            "\x1b[2mthinking\x1b[0m\n"
        );
    }

    #[tokio::test]
    async fn streamed_output_stays_escape_free_when_styled() {
        let mut state = RenderState::styled(true);
        let frame = AgentChunk::new(LLM_OUTPUT, 0, json!({"content": "hello "}));
        let text = rendered_frame(&mut state, &frame, false).await;
        assert_eq!(text, "hello ");
        assert!(!text.contains('\x1b'));
        assert_eq!(state.text, "hello ");
    }

    #[tokio::test]
    async fn answer_usage_and_interaction_frames_ignore_styling() {
        let mut state = RenderState::styled(true);
        let answer = AgentChunk::new(
            ANSWER,
            0,
            json!({"output": "fallback", "result_type": "interrupt"}),
        );
        assert_eq!(
            rendered_frame(&mut state, &answer, false).await,
            "fallback\n"
        );
        assert_eq!(state.answer(), "fallback");
        assert_eq!(state.result_type(), "interrupt");

        let usage = AgentChunk::new(LLM_USAGE, 1, json!({"usage_metadata": {"total_tokens": 5}}));
        assert_eq!(rendered_frame(&mut state, &usage, false).await, "");
        assert_eq!(state.usage, Some(json!({"total_tokens": 5})));

        let interaction = AgentChunk::new(
            INTERACTION,
            2,
            json!({"id": "ask-1", "value": {"question": "q"}}),
        );
        assert_eq!(rendered_frame(&mut state, &interaction, false).await, "");
        assert_eq!(state.interactions.len(), 1);
        assert_eq!(state.interactions[0]["id"], "ask-1");
    }

    #[tokio::test]
    async fn plain_reasoning_is_silent_without_verbose() {
        let mut state = RenderState::styled(true);
        let reasoning = AgentChunk::new(LLM_REASONING, 0, json!({"content": "thinking"}));
        assert_eq!(rendered_frame(&mut state, &reasoning, false).await, "");
    }

    #[tokio::test]
    async fn styled_finish_still_writes_a_plain_newline() {
        let (io, buffer) = scripted_io(&[]);
        let mut state = RenderState::styled(true);
        let frame = AgentChunk::new(LLM_OUTPUT, 0, json!({"content": "answer"}));
        state.render(&frame, Some(&io), false).await.unwrap();
        state.finish(Some(&io)).await.unwrap();
        assert_eq!(output(&buffer), "answer\n");
    }
}
