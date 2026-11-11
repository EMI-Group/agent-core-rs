//! Context windows, token accounting, and compression/offload.
//!
//! Maps to the Python package `core.context_engine` in the openJiuwen reference implementation.

/// Returns the crate name.
pub fn crate_name() -> &'static str {
    "openjiuwen-context"
}
