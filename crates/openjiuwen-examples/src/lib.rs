//! Shared setup helpers for the runnable examples of the Rust SDK.
//!
//! The example binaries in `examples/` each teach one seam of the API, so the
//! setup they have in common lives here instead of being copy-pasted: run-mode
//! detection ([`support::mode`]), subscriber installation ([`support::logging`]),
//! and the stdout narrative helpers ([`support::banner`]).
//!
//! Keeping the helpers in the library (rather than in an example-only module)
//! makes them unit-testable: `cargo test -p openjiuwen-examples` runs these unit
//! tests and then `tests/examples_offline.rs`, which executes every example
//! binary in a clean offline environment. The library has no dev-only dependencies —
//! the scripted model and mock-server fakes live in `openjiuwen-test-support`,
//! which only the example targets may use.
//!
//! Every example starts with [`support::setup`]:
//!
//! ```no_run
//! # fn main() -> openjiuwen::core::common::error::Result<()> {
//! let run_mode = openjiuwen_examples::support::setup()?;
//! openjiuwen_examples::support::banner::note(&format!("mode: {}", run_mode.describe()));
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod support;
