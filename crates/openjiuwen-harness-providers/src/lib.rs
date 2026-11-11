//! Harness provider registry where `native` is implemented and vendor providers are deferred.
//!
//! Maps to the Python package `harness_providers` in the openJiuwen reference implementation.

/// Returns the crate name.
pub fn crate_name() -> &'static str {
    "openjiuwen-harness-providers"
}
