//! Recursive self-improvement (RSI) for the openJiuwen runtime.
//!
//! Maps to the Python package `openjiuwen.rsi` in the openJiuwen reference
//! implementation. The crate owns three things:
//!
//! - [`harness_rsi::auto_harness`] — a bounded, resumable self-optimisation
//!   session loop over candidate configurations.
//! - [`artifact_rsi::program_opt`] — the portable `PUCT` tree search ranking
//!   candidate programs.
//! - [`evolution`] — the shared candidate / mutation / evaluation / selection
//!   protocol used by both halves.
//!
//! Surfaces that need a Python ML runtime, a vendor harness SDK, subprocess or
//! git execution, or distributed evolution live in [`deferred`] as
//! API-shaped stubs returning `Error::Unsupported`.
#![forbid(unsafe_code)]
// Type names deliberately repeat their module name to mirror the Python `rsi` module
// paths (`auto_harness::AutoHarnessConfig`, `program_opt::ProgramOptConfig`, …).
#![allow(clippy::module_name_repetitions)]
#![cfg_attr(
    test,
    allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

pub mod artifact_rsi;
pub mod deferred;
pub mod evolution;
pub mod harness_rsi;

pub use artifact_rsi::program_opt;
pub use harness_rsi::auto_harness;

/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-rsi"
}
