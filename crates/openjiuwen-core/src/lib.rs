//! Kernel umbrella re-exporting the `openjiuwen-core` module tree.
//!
//! Maps to the Python package `core` in the openJiuwen reference implementation.

/// Returns the crate name.
pub fn crate_name() -> &'static str {
    "openjiuwen-core"
}
