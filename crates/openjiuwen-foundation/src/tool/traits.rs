//! The [`Tool`] contract, [`ToolOutput`], and the model-facing render helpers.
//!
//! Ports `openjiuwen/core/foundation/tool/base.py` (`Tool`, `render_payload_text`,
//! `render_tool_output`, `EMPTY_*_TEXT`) and `ToolOutput` from
//! `openjiuwen/core/foundation/tool/schema.py`.
//!
//! A tool is behavior plus a [`ToolCard`]: the implementation may hold transport
//! details (REST configuration, MCP client) that never serialize, while the card
//! stays the model-facing projection.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_common::error::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::card::ToolCard;
use super::context::ToolContext;

/// Placeholder text for a successful call that produced nothing to show.
///
/// Python parity: `EMPTY_SUCCESS_TEXT`.
pub const EMPTY_SUCCESS_TEXT: &str = "Tool succeeded with empty output.";

/// Placeholder text for a failed call that carried no error message.
///
/// Python parity: `EMPTY_FAILURE_TEXT`.
pub const EMPTY_FAILURE_TEXT: &str = "Tool failed without an error message.";

/// The standard structured result returned by [`Tool::invoke`].
///
/// Python parity: `ToolOutput` in `openjiuwen/core/foundation/tool/schema.py`,
/// minus the harness-only `extracted_content`, `include_extracted_content_only_once`
/// and `long_term_memory` fields, which are not part of the Rust surface.
///
/// Unknown fields are ignored, so payloads produced by the Python SDK still
/// deserialize. `success` itself has no serde default on purpose: a plain JSON
/// object that merely looks like a payload (no `success` key) must not be
/// mistaken for a [`ToolOutput`], otherwise the default renderer would report
/// every such result as a failure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ToolOutput {
    /// Whether the call succeeded.
    pub success: bool,
    /// Structured payload; rendered for the model through [`render_payload_text`].
    #[serde(default)]
    pub data: Option<Value>,
    /// Failure message; rendered verbatim for the model when non-empty.
    #[serde(default)]
    pub error: Option<String>,
}

impl ToolOutput {
    /// Creates a successful result carrying `data`.
    #[must_use]
    pub fn success(data: impl Into<Value>) -> Self {
        Self {
            success: true,
            data: Some(data.into()),
            error: None,
        }
    }

    /// Creates a failed result carrying `error`.
    #[must_use]
    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error.into()),
        }
    }
}

/// Renders a `ToolOutput.data` payload as model-facing text.
///
/// Python parity: `render_payload_text`:
/// - `null` renders as the empty string;
/// - an object carrying a `content` key renders only that value (a `null`
///   `content` renders as empty, a string as itself, anything else as compact
///   JSON), so sibling keys stay program-only;
/// - a string renders as itself;
/// - anything else renders as compact JSON.
///
/// Compact JSON never escapes non-ASCII characters, matching Python's
/// `json.dumps(..., ensure_ascii=False)`.
#[must_use]
pub fn render_payload_text(data: &Value) -> String {
    match data {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        Value::Object(map) => match map.get("content") {
            Some(Value::Null) => String::new(),
            Some(Value::String(text)) => text.clone(),
            Some(other) => compact_json(other),
            None => compact_json(data),
        },
        other => compact_json(other),
    }
}

/// Renders a [`ToolOutput`] with the default model-facing rules.
///
/// Python parity: `render_tool_output`. Success renders the payload text, or
/// [`EMPTY_SUCCESS_TEXT`] when that is empty. Failure renders `error` when it is
/// present and non-empty, otherwise the payload text, otherwise
/// [`EMPTY_FAILURE_TEXT`]. An empty-string `error` counts as absent.
#[must_use]
pub fn render_tool_output(output: &ToolOutput) -> String {
    if output.success {
        let rendered = output
            .data
            .as_ref()
            .map_or_else(String::new, render_payload_text);
        if rendered.is_empty() {
            EMPTY_SUCCESS_TEXT.to_owned()
        } else {
            rendered
        }
    } else {
        if let Some(error) = output.error.as_deref().filter(|error| !error.is_empty()) {
            return error.to_owned();
        }
        let rendered = output
            .data
            .as_ref()
            .map_or_else(String::new, render_payload_text);
        if rendered.is_empty() {
            EMPTY_FAILURE_TEXT.to_owned()
        } else {
            rendered
        }
    }
}

/// Serializes `value` compactly, never panicking.
fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// A callable tool: a [`ToolCard`] plus an invocation implementation.
///
/// Python parity: the `Tool` abstract base class in
/// `openjiuwen/core/foundation/tool/base.py`. The trait is object-safe, so
/// callers hold tools as `Arc<dyn Tool>`.
///
/// [`Tool::stream`] takes `self: Arc<Self>` rather than `&self`: it returns a
/// `'static` stream, which is only possible when the tool can be owned by the
/// stream instead of borrowed from the call. The default implementation invokes
/// the tool once and yields that single result.
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// Returns the card describing this tool.
    fn card(&self) -> &ToolCard;

    /// Executes the tool once with already-validated `inputs`.
    ///
    /// # Errors
    ///
    /// Implementations report validation failures as
    /// [`openjiuwen_common::error::Error::Validation`] and execution failures as
    /// [`openjiuwen_common::error::Error::Tool`]; cancellation is
    /// [`openjiuwen_common::error::Error::Cancelled`] (see
    /// [`ToolContext::check_cancelled`]).
    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value>;

    /// Executes the tool and yields incremental results.
    ///
    /// The default invokes the tool once and yields exactly one item. The stream
    /// ends after that item (Python's async generator completion).
    fn stream(
        self: Arc<Self>,
        inputs: Value,
        ctx: ToolContext,
    ) -> BoxStream<'static, Result<Value>> {
        Box::pin(futures::stream::once(async move {
            self.invoke(inputs, &ctx).await
        }))
    }

    /// Renders an [`Tool::invoke`] result into the text the model reads.
    ///
    /// A value that deserializes as a [`ToolOutput`] renders through
    /// [`render_tool_output`]; anything else renders through
    /// [`render_payload_text`].
    ///
    /// Deviation from Python, which renders non-`ToolOutput` results with
    /// `str(output)`: Rust has no equivalent of `str(dict)`/`repr`, and compact
    /// JSON is the closest stable, model-readable form. Strings and objects with
    /// a `content` key render identically to Python.
    fn render_for_llm(&self, output: &Value) -> String {
        match serde_json::from_value::<ToolOutput>(output.clone()) {
            Ok(parsed) => render_tool_output(&parsed),
            Err(_) => render_payload_text(output),
        }
    }

    /// Returns whether this tool may run concurrently with sibling calls.
    ///
    /// Python parity: `Tool.is_parallel_safe` reads `card.parallel_safe`
    /// (defaulting to `true` when the attribute is missing).
    fn is_parallel_safe(&self) -> bool {
        self.card().parallel_safe
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use serde_json::json;

    struct EchoTool {
        card: ToolCard,
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn card(&self) -> &ToolCard {
            &self.card
        }

        async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
            Ok(inputs)
        }
    }

    // port of openjiuwen/core/foundation/tool/base.py render_payload_text

    #[test]
    fn render_payload_text_handles_strings_content_objects_and_null() {
        assert_eq!(render_payload_text(&json!("hello")), "hello");
        assert_eq!(render_payload_text(&json!({"content": "hi"})), "hi");
        assert_eq!(
            render_payload_text(&json!({"content": {"a": 1}})),
            r#"{"a":1}"#
        );
        assert_eq!(render_payload_text(&json!({"content": [1, 2]})), "[1,2]");
        assert_eq!(render_payload_text(&json!({"content": null})), "");
        assert_eq!(render_payload_text(&Value::Null), "");
        assert_eq!(
            render_payload_text(&json!({"a": 1, "b": [1, 2]})),
            r#"{"a":1,"b":[1,2]}"#
        );
        assert_eq!(render_payload_text(&json!(42)), "42");
        assert_eq!(render_payload_text(&json!(true)), "true");
    }

    #[test]
    fn render_payload_text_keeps_non_ascii_unescaped() {
        assert_eq!(render_payload_text(&json!("你好, 世界")), "你好, 世界");
        assert_eq!(render_payload_text(&json!({"content": "你好"})), "你好");
        assert_eq!(
            render_payload_text(&json!({"msg": "你好"})),
            r#"{"msg":"你好"}"#
        );
    }

    // port of openjiuwen/core/foundation/tool/base.py render_tool_output

    #[test]
    fn render_tool_output_matches_python_rules() {
        assert_eq!(
            render_tool_output(&ToolOutput::success(json!({"content": "ok"}))),
            "ok"
        );
        assert_eq!(
            render_tool_output(&ToolOutput::success(json!("plain"))),
            "plain"
        );
        assert_eq!(render_tool_output(&ToolOutput::failure("boom")), "boom");
        assert_eq!(
            render_tool_output(&ToolOutput::success(Value::Null)),
            EMPTY_SUCCESS_TEXT
        );
        assert_eq!(
            render_tool_output(&ToolOutput {
                success: false,
                data: None,
                error: None,
            }),
            EMPTY_FAILURE_TEXT
        );

        let empty_error = ToolOutput {
            success: false,
            data: Some(json!({"content": "payload"})),
            error: Some(String::new()),
        };
        assert_eq!(render_tool_output(&empty_error), "payload");

        let no_error = ToolOutput {
            success: false,
            data: Some(json!({"content": "payload"})),
            error: None,
        };
        assert_eq!(render_tool_output(&no_error), "payload");

        let empty_failure = ToolOutput {
            success: false,
            data: Some(Value::Null),
            error: Some(String::new()),
        };
        assert_eq!(render_tool_output(&empty_failure), EMPTY_FAILURE_TEXT);
    }

    #[test]
    fn tool_output_constructors_and_serde() {
        let success = ToolOutput::success(json!({"content": "ok"}));
        assert!(success.success);
        assert_eq!(success.data, Some(json!({"content": "ok"})));
        assert_eq!(success.error, None);

        let failure = ToolOutput::failure("boom");
        assert!(!failure.success);
        assert_eq!(failure.data, None);
        assert_eq!(failure.error.as_deref(), Some("boom"));

        let json = serde_json::to_value(&success).unwrap();
        assert_eq!(json["success"], true);
        assert_eq!(json["data"]["content"], "ok");
        assert_eq!(json["error"], Value::Null);

        let round_trip: ToolOutput = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip, success);
    }

    #[test]
    fn tool_output_ignores_unknown_python_fields() {
        let parsed: ToolOutput = serde_json::from_value(json!({
            "success": true,
            "data": "ok",
            "error": null,
            "extracted_content": "ignored",
            "include_extracted_content_only_once": false,
            "long_term_memory": "ignored",
        }))
        .unwrap();
        assert_eq!(parsed, ToolOutput::success("ok"));

        assert!(
            serde_json::from_value::<ToolOutput>(json!({"content": "hi"})).is_err(),
            "a plain object without `success` must not read as a ToolOutput"
        );
    }

    #[test]
    fn echo_tool_renders_and_follows_the_card_policy() {
        let tool = EchoTool {
            card: ToolCard::new("echo", "echoes inputs", json!({})),
        };
        assert!(tool.card().parallel_safe);
        assert!(tool.is_parallel_safe());

        assert_eq!(
            tool.render_for_llm(&json!({"success": true, "data": {"content": "done"}})),
            "done"
        );
        assert_eq!(tool.render_for_llm(&json!("plain")), "plain");
        assert_eq!(tool.render_for_llm(&json!({"content": "raw"})), "raw");
        assert_eq!(
            tool.render_for_llm(&json!({"success": false, "error": "nope"})),
            "nope"
        );

        let serial = EchoTool {
            card: ToolCard::new("echo", "", json!({})).with_parallel_safe(false),
        };
        assert!(!serial.is_parallel_safe());
    }

    #[tokio::test]
    async fn default_stream_yields_a_single_invoke_result() {
        let tool: Arc<dyn Tool> = Arc::new(EchoTool {
            card: ToolCard::new("echo", "echoes inputs", json!({})),
        });
        let inputs = json!({"value": 7});
        let expected = tool
            .invoke(inputs.clone(), &ToolContext::new())
            .await
            .unwrap();

        let items: Vec<Result<Value>> = tool
            .clone()
            .stream(inputs, ToolContext::new())
            .collect()
            .await;

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].as_ref().unwrap(), &expected);
    }
}
