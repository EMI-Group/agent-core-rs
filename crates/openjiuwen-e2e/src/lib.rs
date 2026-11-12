//! Cross-crate integration scenarios.
//!
//! Maps to the Python package `openjiuwen.e2e` in the openJiuwen reference implementation.

/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-e2e"
}
