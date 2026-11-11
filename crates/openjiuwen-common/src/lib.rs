//! Shared kernel types for the openJiuwen Rust SDK.
//!
//! This crate maps to the Python package `openjiuwen.core.common` and owns:
//!
//! - [`error`] — the shared [`error::Error`] enum, [`error::Result`], and error context helpers.
//! - [`card`] — the [`card::Card`] trait, [`card::CardBase`], and [`card::CardId`].
//! - [`logging`] — a `tracing`-based logging facade with levels, scopes, and structured events.
//! - [`config`] — environment variable resolvers and `OpenJiuwenHome` path resolution.
//! - [`security`] — [`security::SecretString`], redaction helpers, and constant-time comparison.
//! - [`http`] — a provider-agnostic async HTTP client with retry and streaming support.
//! - [`task`] — a structured async task manager with cancellation and shutdown.
//! - [`utils`] — JSON, id/time, string, and hashing helpers.
//! - [`run_context`] — heartbeat/run context normalization.
//! - [`constants`] — shared protocol constants and controller/task type enums.
//!
//! All I/O is async on tokio, every public data type is `serde`-serializable, and
//! library code never panics on user input.

#![forbid(unsafe_code)]

pub mod card;
pub mod config;
pub mod constants;
pub mod error;
pub mod http;
pub mod logging;
pub mod run_context;
pub mod security;
pub mod task;
pub mod utils;

/// The crate name, useful in diagnostics and documentation.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// The crate version.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
