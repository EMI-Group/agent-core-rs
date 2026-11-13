//! Offline test suite of the `spec` module: parsing, discovery, plugin loading,
//! resolution and the spec-to-`DeepAgent` hand-off.

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod builtin;
mod end_to_end;
mod parse;
mod plugin;
mod registry;
mod resolve;
mod support;
