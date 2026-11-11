//! A2A, OTel tracer, and checkpointer/store/MQ/sysop adapters behind feature gates.
//!
//! Maps to the Python package `extensions` in the openJiuwen reference implementation.

/// Returns the crate name.
pub fn crate_name() -> &'static str {
    "openjiuwen-extensions"
}
