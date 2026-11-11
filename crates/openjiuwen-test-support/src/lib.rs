//! Dev-only shared test infrastructure for the Rust rewrite of openJiuwen.
//!
//! The crate provides a real loopback OpenAI-compatible HTTP server, an
//! in-process scripted `Model`, temp-directory and environment fixtures, JSON
//! assertions, and async eventual-consistency pollers. It replaces the two
//! Python patterns of `tests/unit_tests/fixtures/mock_llm.py`: the scripted
//! `MockLLMModel` (an ordered response queue with no prompt matching) and the
//! `httpx.MockTransport` request fakes. The HTTP fake is a real loopback server
//! so the actual client transport, SSE decoder, and error mapping are exercised
//! instead of stubbed.
//!
//! - `MockLlmServer` — loopback OpenAI-compatible server with a script queue,
//!   request recording, and JSON/SSE wire projection.
//! - `ScriptedModel` — a `Model` implementation with an ordered turn queue for
//!   agent and workflow tests.
//! - [`MockCompletion`] and [`MockToolCall`] — neutral descriptions of an
//!   assistant turn, convertible to `ChatResponse` and to `StreamChunk`
//!   sequences.
//! - `TempWorkspace`, `EnvGuard`, and `TempHomeGuard` — filesystem and
//!   environment isolation.
//! - `assert_json_eq` and friends — JSON equality, subset, and round-trip
//!   assertions with readable diffs.
//! - `eventually` and friends — bounded async polling helpers.
//!
//! The crate is dev-only: it may appear in `[dev-dependencies]` and is used by
//! the `openjiuwen-e2e` scenarios, never by a library crate. Every helper is
//! deterministic (counter-based ids, fixed timestamps, no wall-clock reads) and
//! binds loopback only.
#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::too_many_arguments,
    clippy::items_after_statements,
    clippy::unused_async,
    clippy::significant_drop_tightening,
    clippy::manual_let_else,
    clippy::needless_pass_by_value,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
mod assert_json;
mod eventually;
mod fixtures;
mod mock_llm;
mod responses;
mod scripted_model;
// `assert_json`, `eventually`, `fixtures`, `mock_llm`, and `scripted_model` are
// placeholders filled by parallel work; a glob re-export of an empty module
// trips `unused_imports`/`unreachable_pub`, so the allows stay until they land.
#[allow(unused_imports, unreachable_pub)]
pub use assert_json::*;
#[allow(unused_imports, unreachable_pub)]
pub use eventually::*;
#[allow(unused_imports, unreachable_pub)]
pub use fixtures::*;
#[allow(unused_imports, unreachable_pub)]
pub use mock_llm::*;
pub use responses::*;
#[allow(unused_imports, unreachable_pub)]
pub use scripted_model::*;
pub use wiremock;
/// The crate name, useful in diagnostics and documentation.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
/// The crate version.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
