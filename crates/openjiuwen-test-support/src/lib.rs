//! Mock LLM server, fakes, and fixtures for tests.
//!
//! Maps to the Python package `openjiuwen.test_support` in the openJiuwen reference implementation.

/// Returns the crate name.
pub fn crate_name() -> &'static str {
    "openjiuwen-test-support"
}
