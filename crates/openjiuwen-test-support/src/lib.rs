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
//! Every module is public, so each item is also reachable at its module path
//! (`openjiuwen_test_support::mock_llm::MockLlmServer`,
//! `openjiuwen_test_support::fixtures::TempWorkspace`) while the crate root
//! glob-re-exports the same items for flat imports.
//!
//! The crate is dev-only: it may appear in `[dev-dependencies]` and is used by
//! the `openjiuwen-e2e` scenarios, never by a library crate. Every helper is
//! deterministic (counter-based ids, fixed timestamps, no wall-clock reads) and
//! binds loopback only.
#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
// Dev-only test infrastructure, so the ergonomic allowances below are inherited
// from the sibling library crates (`crates/openjiuwen-foundation/src/lib.rs`)
// rather than paid for with ceremony no test author benefits from:
// - `missing_errors_doc`/`missing_panics_doc`: every helper that can fail returns
//   a documented `Result` on a short signature; the docs live on the module.
// - `must_use_candidate`/`module_name_repetitions`/`too_many_arguments`/
//   `items_after_statements`/`unused_async`/`significant_drop_tightening`/
//   `manual_let_else`/`needless_pass_by_value`: wire projection and fixture
//   ergonomics — builder chains, short synchronous lock sections, and by-value
//   `impl Into<String>` parameters stay readable without extra annotations.
// - `cast_possible_truncation`/`cast_sign_loss`: deterministic counter and
//   wire-index casts bounded by the number of scripted turns.
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
pub mod assert_json;
pub mod eventually;
pub mod fixtures;
// Crate-private shared constants and the poison-recovering mutex helper; its
// items are `pub(crate)` and intentionally not re-exported as public API.
mod internal;
pub mod mock_llm;
pub mod responses;
pub mod scripted_model;
pub use assert_json::*;
pub use eventually::*;
pub use fixtures::*;
pub use mock_llm::*;
pub use responses::*;
pub use scripted_model::*;
pub use wiremock;
/// The crate name, useful in diagnostics and documentation.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
/// The crate version.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
