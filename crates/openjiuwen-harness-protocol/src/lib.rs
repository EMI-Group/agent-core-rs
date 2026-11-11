//! Provider-neutral harness SPI with traits, events, results, and streams.
//!
//! Maps to the Python package `harness_protocol` in the openJiuwen reference implementation.

/// Returns the crate name.
pub fn crate_name() -> &'static str {
    "openjiuwen-harness-protocol"
}
