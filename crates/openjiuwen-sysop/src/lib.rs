//! Local and sandbox filesystem, shell, and code operations.
//!
//! Maps to the Python package `core.sys_operation` in the openJiuwen reference implementation.

/// Returns the crate name.
pub fn crate_name() -> &'static str {
    "openjiuwen-sysop"
}
