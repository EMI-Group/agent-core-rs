//! Re-export shim over `rsi::auto_harness`.
//!
//! Maps to the Python package `rsi.auto_harness` in the openJiuwen reference implementation.

/// Returns the crate name.
pub fn crate_name() -> &'static str {
    "openjiuwen-auto-harness"
}
