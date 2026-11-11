//! Agent builder, prompt builder, skill creator/evaluator, and tuning helpers.
//!
//! Maps to the Python package `dev_tools` in the openJiuwen reference implementation.

/// Returns the crate name.
pub fn crate_name() -> &'static str {
    "openjiuwen-dev-tools"
}
