//! Sessions, state scopes, streaming, interactions, checkpointers, and tracer.
//!
//! Maps to the Python package `core.session` in the openJiuwen reference implementation.

/// Returns the crate name.
pub fn crate_name() -> &'static str {
    "openjiuwen-session"
}
