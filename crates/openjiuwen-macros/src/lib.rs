//! `#[tool]` — the Rust equivalent of the Python `@tool` decorator.
//!
//! One attribute turns an `async fn` plus a typed argument struct into a
//! model-facing `ToolCard`, a `Tool` implementation, and an
//! `Arc<dyn Tool>` constructor, so a tool author writes a plain async function
//! and never hand-builds a card or a JSON Schema.
//!
//! # Supported forms
//!
//! ```
//! use openjiuwen_common::error::Result;
//! use openjiuwen_foundation::tool::{Tool, ToolContext};
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(Debug, Deserialize, JsonSchema)]
//! pub struct AddArgs {
//!     pub a: i64,
//!     pub b: i64,
//! }
//!
//! /// Add two integers.
//! #[openjiuwen_macros::tool(name = "add")]
//! pub async fn add(args: AddArgs) -> Result<i64> {
//!     Ok(args.a + args.b)
//! }
//!
//! // Optional second parameter: `ctx: &ToolContext` or `ctx: ToolContext`.
//! // The return type may also be a plain `T` instead of `Result<T>`.
//!
//! // The expansion keeps `add` callable and adds `AddTool` + `add_tool()`.
//! let tool = add_tool();
//! assert_eq!(tool.card().name, "add");
//! assert_eq!(tool.card().description, "Add two integers.");
//! assert!(tool.card().input_params["properties"]["a"].is_object());
//!
//! # let runtime = tokio::runtime::Builder::new_current_thread()
//! #     .enable_all()
//! #     .build()
//! #     .unwrap();
//! let output = runtime
//!     .block_on(tool.invoke(serde_json::json!({"a": 2, "b": 3}), &ToolContext::new()))
//!     .unwrap();
//! assert_eq!(output, serde_json::json!(5));
//! ```
//!
//! The expansion keeps the annotated function unchanged and adds `AddTool`
//! (a wrapper around `FunctionTool`), `AddTool::NAME` / `AddTool::DESCRIPTION`,
//! `AddTool::new()`, an `impl Tool for AddTool`, and `add_tool()`, which returns
//! `Arc<dyn Tool>`.
//!
//! # Attributes
//!
//! `#[tool(name = "…", description = "…", input_params = <expr>, stateless =
//! true, parallel_safe = false, idempotent = true, exposure = "deferred",
//! execution = "client", crate = "::openjiuwen_foundation", common_crate =
//! "::openjiuwen_common")]` — every key is optional; the defaults mirror
//! Python's tool card defaults. Without `description`, the first doc-comment
//! paragraph is used, falling back to the humanized function name.
//!
//! # What a user crate needs
//!
//! The generated code resolves `openjiuwen_foundation` (the `Tool` surface),
//! `openjiuwen_common` (the error type), `serde_json`, and `async-trait`; the
//! argument struct needs `serde::Deserialize` and `schemars::JsonSchema` unless
//! `input_params` supplies the schema. Crates that re-export the foundation
//! surface under another path can point `crate`/`common_crate` at it.
//!
//! The macro itself is compile-time only: it never runs, allocates, or logs at
//! runtime, and it reports every problem as a spanned `compile_error!`.

extern crate proc_macro;

use proc_macro::TokenStream;

mod describe;
mod error;
mod naming;
mod tool;

/// Turns an `async fn` into a model-facing tool.
///
/// See the [crate documentation](crate) for the supported forms and attribute
/// keys. On error the original item is preserved next to a spanned
/// `compile_error!`, so the compiler reports only the real problem.
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    tool::expand(attr.into(), item.into()).into()
}
