//! Function-backed tools: [`FunctionTool`], [`FunctionToolBuilder`], and the
//! boxed handler type.
//!
//! Ports `LocalFunction` from
//! `openjiuwen/core/foundation/tool/function/function.py`:
//! - Python `invoke` formats the arguments through
//!   `SchemaUtils.format_with_schema` (validate + fill defaults) and then calls
//!   `func(**inputs)`; [`FunctionTool::invoke`] does the same with
//!   [`ensure_object_schema`], [`apply_defaults`], and [`validate_inputs`], then
//!   calls the handler with the validated inputs as a single `Value`.
//! - Python `render_for_llm` delegates to the optional `render` callable;
//!   [`FunctionTool::render_for_llm`] delegates to the [`FunctionToolBuilder::render`]
//!   override and otherwise falls back to the [`Tool::render_for_llm`] default
//!   (`render_tool_output` for a [`ToolOutput`], `render_payload_text` otherwise).
//! - Python's generator `stream` is not ported: [`Tool::stream`] is inherited
//!   (one `invoke` result), so a function tool is uniformly a single-shot call.
//!
//! **Handlers are async-only.** Python accepts blocking `def` functions and calls
//! them inline; a Rust handler returns a future, so a blocking or CPU-heavy body
//! must move its work into `tokio::task::spawn_blocking` *inside* the handler
//! instead of blocking the task that awaits the tool.
//!
//! # The `#[tool]` macro contract
//!
//! The `openjiuwen-macros` `#[tool]` attribute does **not** emit the fallible
//! builder chain. It generates a wrapper struct holding a [`FunctionTool`] whose
//! `new()` builds the card and the boxed handler directly, then wraps them with
//! [`FunctionTool::new`] (infallible, so no `Result` is involved). The generated
//! code is equivalent to:
//!
//! ```ignore
//! // `Args` derives `serde::Deserialize + schemars::JsonSchema`
//! let card = ToolCard::new(Self::NAME, Self::DESCRIPTION, schema_for::<Args>())
//!     /* .with_* policy chain for the attributes that were supplied */;
//! let handler: ToolHandler = Arc::new(|inputs: Value, ctx: ToolContext| {
//!     Box::pin(async move { /* deserialize `Args`, call the fn, serialize the output */ })
//! });
//! Self { inner: FunctionTool::new(card, handler) }
//! ```
//!
//! The wrapper's `Tool` impl is annotated with `#[::async_trait::async_trait]`,
//! so a consumer crate whose tools are macro-generated needs `async-trait` in
//! its own dependencies (the absolute `::async_trait::` path does not resolve
//! through this crate). Hand-written `Tool` (or `Model`) impls can instead use
//! the crate-root re-export, `#[openjiuwen_foundation::async_trait::async_trait]`,
//! and skip the extra dependency.
//!
//! [`FunctionTool::builder`] remains the way to build a tool by hand; its
//! [`FunctionToolBuilder::build`] returns [`Result`], which is why the macro
//! avoids it.
//!
//! ```
//! use openjiuwen_foundation::tool::function::FunctionTool;
//! use openjiuwen_foundation::tool::traits::Tool;
//!
//! let tool = FunctionTool::builder()
//!     .name("echo")
//!     .description("echoes its input")
//!     .handler(|inputs, _ctx| async move { Ok(inputs) })
//!     .build()
//!     .unwrap();
//!
//! assert_eq!(tool.card().name, "echo");
//! ```

use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use openjiuwen_common::error::{Error, Result};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};

use super::card::ToolCard;
use super::context::ToolContext;
use super::schema::{apply_defaults, ensure_object_schema, schema_for, validate_inputs};
use super::traits::{render_payload_text, render_tool_output, Tool, ToolOutput};

/// Boxed async handler: validated, default-filled inputs + invocation context ->
/// result.
///
/// The handler receives the arguments **after** [`ensure_object_schema`],
/// [`apply_defaults`], and [`validate_inputs`] ran, so it never needs to
/// re-validate them. Its return value is passed through verbatim by
/// [`FunctionTool::invoke`]: a string, an object, an array, or a
/// [`ToolOutput`] are all legal.
pub type ToolHandler =
    Arc<dyn Fn(Value, ToolContext) -> BoxFuture<'static, Result<Value>> + Send + Sync>;

/// A custom model-facing renderer installed through
/// [`FunctionToolBuilder::render`].
type RenderHandler = Arc<dyn Fn(&Value) -> String + Send + Sync>;

/// A tool backed by a Rust closure (Python `LocalFunction`).
///
/// The [`ToolCard`] is the model-facing projection; the handler and the optional
/// renderer are runtime-only and never serialize.
///
/// Build one with [`FunctionTool::builder`] (the `#[tool]` macro target) or
/// directly from a card and a [`ToolHandler`] with [`FunctionTool::new`].
pub struct FunctionTool {
    card: ToolCard,
    handler: ToolHandler,
    render: Option<RenderHandler>,
}

impl FunctionTool {
    /// Creates a function tool from an already-built `card` and `handler`.
    ///
    /// The handler is trusted: nothing validates it, and [`FunctionTool::invoke`]
    /// still validates and default-fills the inputs against `card.input_params`
    /// before every call.
    #[must_use]
    pub fn new(card: ToolCard, handler: ToolHandler) -> Self {
        Self {
            card,
            handler,
            render: None,
        }
    }

    /// Returns a fresh [`FunctionToolBuilder`].
    #[must_use]
    pub fn builder() -> FunctionToolBuilder {
        FunctionToolBuilder::default()
    }

    /// Returns this tool as an `Arc<dyn Tool>` so it can be stored in tool
    /// registries alongside other implementations.
    #[must_use]
    pub fn boxed(self) -> Arc<dyn Tool> {
        Arc::new(self)
    }
}

impl std::fmt::Debug for FunctionTool {
    /// Prints the card and whether a custom renderer is installed; the handler
    /// itself is opaque and never formatted.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FunctionTool")
            .field("card", &self.card)
            .field("custom_render", &self.render.is_some())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Tool for FunctionTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    /// Validates and default-fills `inputs`, then runs the handler.
    ///
    /// Python parity: `LocalFunction.invoke`. Validation failures are
    /// [`Error::Validation`]; cancellation is [`Error::Cancelled`]; the handler's
    /// own error is returned unchanged. The handler is never called when
    /// validation fails or the context is already cancelled.
    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        let schema = ensure_object_schema(self.card.input_params.clone());
        let mut inputs = inputs;
        apply_defaults(&schema, &mut inputs);
        validate_inputs(&schema, &inputs)?;

        ctx.check_cancelled()?;

        (self.handler)(inputs, ctx.clone()).await
    }

    /// Renders through the [`FunctionToolBuilder::render`] override when one was
    /// supplied, otherwise through the [`Tool::render_for_llm`] default.
    fn render_for_llm(&self, output: &Value) -> String {
        match &self.render {
            Some(render) => render(output),
            None => default_render(output),
        }
    }
}

impl From<FunctionTool> for Arc<dyn Tool> {
    fn from(tool: FunctionTool) -> Self {
        tool.boxed()
    }
}

/// Reproduces the [`Tool::render_for_llm`] default for [`FunctionTool`].
///
/// Rust cannot call an overridden trait method's default body, so the rules live
/// here too: a [`ToolOutput`] renders through [`render_tool_output`], anything
/// else through [`render_payload_text`]. Keep the two in sync.
fn default_render(output: &Value) -> String {
    match serde_json::from_value::<ToolOutput>(output.clone()) {
        Ok(parsed) => render_tool_output(&parsed),
        Err(_) => render_payload_text(output),
    }
}

/// Builder for [`FunctionTool`].
///
/// Both handler shapes are supported:
/// - [`FunctionToolBuilder::handler`] takes a closure over the validated
///   [`Value`] inputs;
/// - [`FunctionToolBuilder::typed_handler`] deserializes the inputs into `I` and
///   serializes the `O` output, remembering `I` so [`FunctionToolBuilder::build`]
///   can derive the input schema when the caller set none.
///
/// A literal schema ([`FunctionToolBuilder::input_schema`]) or a derived one
/// ([`FunctionToolBuilder::input_schema_for`]) always wins over the schema
/// inferred from a typed handler, so the `#[tool]` macro's explicit
/// `input_schema(schema_for::<Args>())` call is authoritative.
#[derive(Default)]
pub struct FunctionToolBuilder {
    name: String,
    description: String,
    input_params: Option<Value>,
    derived_schema: Option<fn() -> Value>,
    handler: Option<ToolHandler>,
    render: Option<RenderHandler>,
}

impl FunctionToolBuilder {
    /// Creates an empty builder; [`FunctionToolBuilder::build`] fails without a
    /// handler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the tool name the model sees.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Sets the human-readable description shown to the model.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Sets a literal JSON Schema for the accepted arguments.
    ///
    /// The schema is normalized by [`ensure_object_schema`] on every
    /// [`FunctionTool::invoke`], so a non-object schema is tolerated here.
    #[must_use]
    pub fn input_schema(mut self, schema: Value) -> Self {
        self.input_params = Some(schema);
        self
    }

    /// Derives the argument schema from `T` through [`schema_for`].
    ///
    /// Pair it with [`FunctionToolBuilder::typed_handler`] whose input type is
    /// `T`; naming the type at the schema call site keeps the compile-time link
    /// explicit, which is why the `#[tool]` macro always calls it.
    #[must_use]
    pub fn input_schema_for<T: JsonSchema>(mut self) -> Self {
        self.input_params = Some(schema_for::<T>());
        self
    }

    /// Sets a closure-based handler that receives the validated inputs.
    ///
    /// The handler's return value is returned verbatim by
    /// [`FunctionTool::invoke`]; its errors are propagated unchanged.
    #[must_use]
    pub fn handler<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(Value, ToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        self.handler = Some(Arc::new(move |inputs: Value, ctx: ToolContext| {
            Box::pin(handler(inputs, ctx)) as BoxFuture<'static, Result<Value>>
        }));
        self
    }

    /// Sets a typed handler: the validated inputs are deserialized into `I` and
    /// the returned `O` is serialized back to JSON.
    ///
    /// Input deserialization failures are [`Error::Validation`], output
    /// serialization failures are [`Error::Tool`]. `I` is remembered, so
    /// [`FunctionToolBuilder::build`] derives the argument schema with
    /// [`schema_for`] when neither
    /// [`FunctionToolBuilder::input_schema`] nor
    /// [`FunctionToolBuilder::input_schema_for`] was called; the `#[tool]` macro
    /// still sets the schema explicitly, and that explicit schema wins.
    ///
    /// The type parameters are ordered `<I, O, F, Fut>` so call sites write
    /// `typed_handler::<Input, Output, _, _>(handler)`.
    #[must_use]
    pub fn typed_handler<I, O, F, Fut>(mut self, handler: F) -> Self
    where
        I: DeserializeOwned + JsonSchema + Send + 'static,
        O: Serialize + Send + 'static,
        F: Fn(I, ToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O>> + Send + 'static,
    {
        let handler = Arc::new(handler);
        self.derived_schema = Some(schema_for::<I>);
        self.handler = Some(Arc::new(move |inputs: Value, ctx: ToolContext| {
            let handler = Arc::clone(&handler);
            Box::pin(async move {
                let typed: I = serde_json::from_value(inputs)
                    .map_err(|error| Error::validation(format!("invalid tool input: {error}")))?;
                let output = handler(typed, ctx).await?;
                serde_json::to_value(output).map_err(|error| {
                    Error::tool(format!("failed to serialize tool output: {error}"))
                })
            }) as BoxFuture<'static, Result<Value>>
        }));
        self
    }

    /// Sets a custom renderer producing the model-facing text of a result.
    ///
    /// Python parity: `LocalFunction(..., render=...)`; it replaces the
    /// [`Tool::render_for_llm`] default for non-[`ToolOutput`] results too.
    #[must_use]
    pub fn render<F>(mut self, render: F) -> Self
    where
        F: Fn(&Value) -> String + Send + Sync + 'static,
    {
        self.render = Some(Arc::new(render));
        self
    }

    /// Builds the tool.
    ///
    /// The argument schema is the one set explicitly, else the one inferred from
    /// a typed handler, else `{}` (`ToolCard`'s unset form). `build` returns
    /// [`Result`] because the handler is mandatory, which is why the `#[tool]`
    /// macro emits the builder chain inside a function returning
    /// `Result<FunctionTool>`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when no handler was set.
    pub fn build(self) -> Result<FunctionTool> {
        let handler = self
            .handler
            .ok_or_else(|| Error::validation("function tool requires a handler"))?;
        let input_params = self
            .input_params
            .or_else(|| self.derived_schema.map(|schema| schema()))
            .unwrap_or_else(|| json!({}));

        Ok(FunctionTool {
            card: ToolCard::new(self.name, self.description, input_params),
            handler,
            render: self.render,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use futures::StreamExt;
    use openjiuwen_common::error::ErrorKind;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    use super::*;
    use crate::tool::traits::{EMPTY_FAILURE_TEXT, EMPTY_SUCCESS_TEXT};

    /// Arguments with a required field and a defaulted one.
    fn greet_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "mode": {"type": "string", "default": "fast"}
            },
            "required": ["name"]
        })
    }

    #[tokio::test]
    async fn closure_handler_receives_validated_default_filled_inputs() {
        let observed = Arc::new(Mutex::new(None));
        let recorder = Arc::clone(&observed);

        let tool = FunctionTool::builder()
            .name("greet")
            .description("greets a name")
            .input_schema(greet_schema())
            .handler(move |inputs: Value, ctx: ToolContext| {
                let recorder = Arc::clone(&recorder);
                async move {
                    *recorder.lock().unwrap() = Some(json!({
                        "inputs": inputs,
                        "session": ctx.session_id,
                    }));
                    Ok(inputs)
                }
            })
            .build()
            .unwrap();

        let ctx = ToolContext::new().with_session("session-42");
        let result = tool.invoke(json!({"name": "bob"}), &ctx).await.unwrap();

        assert_eq!(result, json!({"name": "bob", "mode": "fast"}));
        let observed = observed.lock().unwrap().clone().unwrap();
        assert_eq!(observed["inputs"], json!({"name": "bob", "mode": "fast"}));
        assert_eq!(observed["session"], json!("session-42"));

        // the handler's return value is passed through verbatim
        for payload in [json!("plain text"), json!({"a": 1}), json!([1, 2, 3])] {
            let value = payload.clone();
            let tool = FunctionTool::builder()
                .handler(move |_inputs: Value, _ctx: ToolContext| {
                    let value = value.clone();
                    async move { Ok(value) }
                })
                .build()
                .unwrap();

            let result = tool.invoke(json!({}), &ToolContext::new()).await.unwrap();
            assert_eq!(result, payload);
        }
    }

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct AddArgs {
        a: i64,
        b: i64,
    }

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct AddResult {
        sum: i64,
    }

    #[tokio::test]
    async fn typed_handler_round_trips_typed_input_and_output() {
        let tool = FunctionTool::builder()
            .name("add")
            .description("adds two numbers")
            .typed_handler::<AddArgs, AddResult, _, _>(|args, _ctx| async move {
                Ok(AddResult {
                    sum: args.a + args.b,
                })
            })
            .build()
            .unwrap();

        assert_eq!(
            tool.card().input_params["properties"]["a"]["type"],
            "integer"
        );

        let result = tool
            .invoke(json!({"a": 2, "b": 3}), &ToolContext::new())
            .await
            .unwrap();
        assert_eq!(result, json!({"sum": 5}));

        let wrong_type = tool
            .invoke(json!({"a": "two", "b": 3}), &ToolContext::new())
            .await
            .unwrap_err();
        assert_eq!(wrong_type.kind(), ErrorKind::Validation);
    }

    #[tokio::test]
    async fn invalid_inputs_fail_validation_without_calling_the_handler() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);

        let tool = FunctionTool::builder()
            .input_schema(greet_schema())
            .handler(move |inputs: Value, _ctx: ToolContext| {
                let counter = Arc::clone(&counter);
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Ok(inputs)
                }
            })
            .build()
            .unwrap();

        let error = tool
            .invoke(json!({"name": 7}), &ToolContext::new())
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(matches!(error, Error::Validation(_)));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn handler_errors_propagate_unchanged() {
        let tool = FunctionTool::builder()
            .handler(|_inputs: Value, _ctx: ToolContext| async move {
                Err(Error::Tool("boom".to_owned()))
            })
            .build()
            .unwrap();

        let error = tool
            .invoke(json!({}), &ToolContext::new())
            .await
            .unwrap_err();

        assert!(
            matches!(&error, Error::Tool(message) if message == "boom"),
            "the handler error must come back unchanged, got {error:?}"
        );
        assert_eq!(error.kind(), ErrorKind::Tool);
    }

    #[tokio::test]
    async fn cancelled_context_fails_without_calling_the_handler() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);

        let tool = FunctionTool::builder()
            .handler(move |inputs: Value, _ctx: ToolContext| {
                let counter = Arc::clone(&counter);
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Ok(inputs)
                }
            })
            .build()
            .unwrap();

        let ctx = ToolContext::new();
        ctx.cancellation.cancel();

        let error = tool.invoke(json!({}), &ctx).await.unwrap_err();

        assert!(matches!(error, Error::Cancelled));
        assert_eq!(error.kind(), ErrorKind::Cancelled);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn render_override_wins_and_default_render_follows_tool_rules() {
        let custom = FunctionTool::builder()
            .render(|output| format!("custom:{output}"))
            .handler(|inputs: Value, _ctx: ToolContext| async move { Ok(inputs) })
            .build()
            .unwrap();
        assert_eq!(
            custom.render_for_llm(&json!({"success": true, "data": "ok"})),
            r#"custom:{"success":true,"data":"ok"}"#
        );

        let default = FunctionTool::builder()
            .handler(|inputs: Value, _ctx: ToolContext| async move { Ok(inputs) })
            .build()
            .unwrap();

        // a ToolOutput renders through the default ToolOutput rules
        assert_eq!(
            default.render_for_llm(&json!({"success": true, "data": {"content": "ok"}})),
            "ok"
        );
        assert_eq!(
            default.render_for_llm(&json!({"success": false, "error": "nope"})),
            "nope"
        );
        assert_eq!(
            default.render_for_llm(&json!({"success": true})),
            EMPTY_SUCCESS_TEXT
        );
        assert_eq!(
            default.render_for_llm(&json!({"success": false})),
            EMPTY_FAILURE_TEXT
        );

        // a plain payload renders through render_payload_text
        assert_eq!(default.render_for_llm(&json!({"a": 1})), r#"{"a":1}"#);
        assert_eq!(default.render_for_llm(&json!({"content": "raw"})), "raw");
        assert_eq!(default.render_for_llm(&json!("plain")), "plain");
    }

    #[tokio::test]
    async fn default_stream_yields_one_item_equal_to_invoke() {
        let tool: Arc<dyn Tool> = FunctionTool::builder()
            .name("echo")
            .handler(|inputs: Value, _ctx: ToolContext| async move { Ok(inputs) })
            .build()
            .unwrap()
            .boxed();

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

    #[test]
    fn build_requires_a_handler_and_derives_typed_schemas() {
        let error = FunctionTool::builder()
            .name("no-handler")
            .build()
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);

        let explicit = FunctionTool::builder()
            .input_schema_for::<AddArgs>()
            .handler(|inputs: Value, _ctx: ToolContext| async move { Ok(inputs) })
            .build()
            .unwrap();
        assert_eq!(explicit.card().input_params["type"], "object");
        assert!(explicit.card().input_params["properties"]["a"].is_object());
        assert!(explicit.card().input_params["properties"]["b"].is_object());

        // typed_handler without any explicit schema derives it from `I`
        let derived = FunctionTool::builder()
            .typed_handler::<AddArgs, AddResult, _, _>(|args, _ctx| async move {
                Ok(AddResult {
                    sum: args.a + args.b,
                })
            })
            .build()
            .unwrap();
        assert_eq!(derived.card().input_params["type"], "object");
        assert_eq!(derived.card().input_params["required"], json!(["a", "b"]));

        // the explicit schema wins over the one inferred from a typed handler
        let authoritative = FunctionTool::builder()
            .typed_handler::<AddArgs, AddResult, _, _>(|args, _ctx| async move {
                Ok(AddResult {
                    sum: args.a + args.b,
                })
            })
            .input_schema_for::<AddResult>()
            .build()
            .unwrap();
        assert!(authoritative.card().input_params["properties"]
            .get("sum")
            .is_some());

        // a builder without a schema keeps the unset `{}` form
        let untyped = FunctionTool::builder()
            .handler(|inputs: Value, _ctx: ToolContext| async move { Ok(inputs) })
            .build()
            .unwrap();
        assert_eq!(untyped.card().input_params, json!({}));
    }

    #[tokio::test]
    async fn card_and_parallel_safe_follow_the_card() {
        let card = ToolCard::new("calc", "does math", json!({})).with_parallel_safe(false);
        let tool = FunctionTool::new(
            card.clone(),
            Arc::new(|inputs: Value, _ctx: ToolContext| {
                Box::pin(async move { Ok(inputs) }) as BoxFuture<'static, Result<Value>>
            }),
        );

        assert_eq!(tool.card(), &card);
        assert!(!tool.is_parallel_safe());

        let boxed: Arc<dyn Tool> = tool.into();
        assert_eq!(boxed.card().name, "calc");
        assert!(!boxed.is_parallel_safe());
        assert_eq!(
            boxed
                .invoke(json!({"x": 1}), &ToolContext::new())
                .await
                .unwrap(),
            json!({"x": 1})
        );

        let parallel = FunctionTool::builder()
            .name("parallel")
            .handler(|inputs: Value, _ctx: ToolContext| async move { Ok(inputs) })
            .build()
            .unwrap();
        assert!(parallel.card().parallel_safe);
        assert!(parallel.is_parallel_safe());
    }
}
