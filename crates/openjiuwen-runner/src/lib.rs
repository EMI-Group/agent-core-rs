//! `Runner`, `ResourceMgr`, callbacks, and the MQ abstraction.
//!
//! Maps to the Python package `core.runner` in the openJiuwen reference implementation.

/// Returns the crate name.
pub fn crate_name() -> &'static str {
    "openjiuwen-runner"
}
