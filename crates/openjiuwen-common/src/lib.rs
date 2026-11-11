//! Card/Config schema, typed errors, logging, env/config, HTTP helpers, task manager, and utilities.
//!
//! Maps to the Python package `core.common` in the openJiuwen reference implementation.

/// Returns the crate name.
pub fn crate_name() -> &'static str {
    "openjiuwen-common"
}
