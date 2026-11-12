//! Shared helpers for the twelve example binaries.
//!
//! An example reaches this module with `mod common;` (they all sit in
//! `examples/`, so the file is `examples/common/mod.rs`) and then calls
//! [`setup`] before its own body.
//!
//! Unlike the library in `src/`, this module is only compiled into example
//! targets, so it may use dev-dependencies (`openjiuwen-test-support`,
//! `tempfile`) and `anyhow`: the fakes that keep an example offline belong
//! here, and `anyhow` keeps an example's `main` readable. The library must never
//! use either.
//!
//! # Unused helpers
//!
//! A shared module included by twelve binaries is always partially unused, so
//! every item carries `#[allow(dead_code)]` with the reason: a given example
//! uses the scripted model *or* the mock server, never both, and the helpers it
//! does not call would otherwise produce a warning per binary.

#![allow(
    // This module is compiled into a binary crate, which nothing can link
    // against, so `pub` items are unreachable by definition. They stay `pub`
    // because that is how the library-shaped SDK surface reads and because the
    // item list is meant to be copied into a real example verbatim.
    unreachable_pub,
    // Example `main`s and helpers are ergonomic wrappers: the error cases they
    // add (`# Errors` docs, panics) are documented at the called SDK API, and a
    // helper that always succeeds would only make the call sites noisier.
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // Helpers return values purely for the caller's convenience.
    clippy::must_use_candidate,
    // `skip` and `setup` always return `Ok`, but mirror the SDK signatures so an
    // example can use `?` uniformly.
    clippy::unnecessary_wraps
)]

use std::sync::Arc;

use openjiuwen::core::foundation::llm::{init_model, Model, ModelClientConfig, ModelRequestConfig};
use openjiuwen_test_support::{
    MockCompletion, MockLlmServer, ScriptedModel, ScriptedResponse, ScriptedTurn,
};

// Re-exported so an example needs a single `use common::...` line for its
// banners, its run-mode type, and the `RunMode` match it does on the result of
// `setup`.
#[allow(unused_imports)]
pub use openjiuwen_examples::support::{banner, mode, RunMode};

/// Runs the library setup, adapted to `anyhow`.
///
/// Loads `.env`, installs stderr logging, prints the banner, and returns the
/// detected [`RunMode`]. Prefer this over calling
/// `openjiuwen_examples::support::setup` directly so the rest of the example can
/// use `anyhow::Result`.
#[allow(dead_code)]
pub fn setup() -> anyhow::Result<RunMode> {
    Ok(openjiuwen_examples::support::setup()?)
}

/// An offline model that answers with `answer` on its single call.
///
/// The default for examples that exercise one model round: the scripted model is
/// in-process, deterministic, and needs no network.
#[allow(dead_code)]
pub fn scripted(answer: impl Into<String>) -> Arc<dyn Model> {
    Arc::new(ScriptedModel::text(answer))
}

/// An offline model that asks for `tool` with `arguments`, then answers.
///
/// The first turn carries the tool call an agent loop is expected to satisfy;
/// the second is the final answer produced once the tool result is fed back.
#[allow(dead_code)]
pub fn scripted_tool_round(
    tool: &str,
    arguments: serde_json::Value,
    answer: impl Into<String>,
) -> Arc<dyn Model> {
    let call = MockCompletion::tool_call(tool, arguments).to_chat_response("");
    Arc::new(ScriptedModel::new([
        ScriptedTurn::Response(call),
        ScriptedTurn::text(answer),
    ]))
}

/// Turns a hand-built [`ScriptedModel`] into the `Arc<dyn Model>` an agent takes.
///
/// Use this when an example needs more control than [`scripted`] or
/// [`scripted_tool_round`] offer, for example to keep a clone for inspecting
/// `calls()` afterwards.
#[allow(dead_code)]
pub fn arc_model(model: ScriptedModel) -> Arc<dyn Model> {
    Arc::new(model)
}

/// One plain text answer for a [`MockLlmServer`] script.
///
/// The mock server speaks the real OpenAI-compatible HTTP protocol, so a script
/// entry is a whole response rather than a model turn.
#[allow(dead_code)]
pub fn text_response(text: impl Into<String>) -> ScriptedResponse {
    ScriptedResponse::text(text)
}

/// Starts a loopback mock LLM server scripted with `items`, in order.
///
/// Use it when the wire path matters — a component that builds its own client
/// from a base URL — and point that component at [`MockLlmServer::base_url`].
/// The server shuts down when the returned handle is dropped.
#[allow(dead_code)]
pub async fn mock_server(items: Vec<ScriptedResponse>) -> anyhow::Result<MockLlmServer> {
    Ok(MockLlmServer::builder().script(items).try_start().await?)
}

/// Builds a real provider client from the environment, for the live branch only.
///
/// Reads the provider, credential, and base URL from `MODEL_PROVIDER`,
/// `OPENJIUWEN_API_KEY`/`API_KEY`/`OPENAI_API_KEY`, and
/// `API_BASE`/`OPENAI_BASE_URL`, and the model name from `MODEL_NAME`. Call it
/// only after [`setup`] reported a live [`RunMode`]; in offline mode the
/// environment has no credential and this returns an error.
#[allow(dead_code)]
pub async fn live_model() -> anyhow::Result<Arc<dyn Model>> {
    let client = ModelClientConfig::from_env();
    let request = ModelRequestConfig::from_env();
    Ok(init_model(client, request).await?)
}

/// A fresh temporary directory, removed when the returned handle is dropped.
///
/// Examples that need a workspace, a store, or a checkpoint path keep the
/// `TempDir` alive for their whole run so nothing is written outside it.
#[allow(dead_code)]
pub fn tempdir() -> anyhow::Result<tempfile::TempDir> {
    Ok(tempfile::tempdir()?)
}

/// Prints why an example cannot run its main path, and how to enable it.
///
/// Returns `Ok(())` so an example can `return common::skip(...)` and still exit
/// successfully: a missing optional prerequisite is a documented outcome, not a
/// failure.
#[allow(dead_code)]
pub fn skip(reason: &str, hint: &str) -> anyhow::Result<()> {
    banner::note(reason);
    banner::note(&format!("set {hint} to run this example"));
    Ok(())
}
