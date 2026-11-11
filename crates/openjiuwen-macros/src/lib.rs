//! Proc macros for `#[tool]` and schema derivation helpers.
//!
//! Maps to the Python package `core.foundation.tools` in the openJiuwen reference implementation.

use proc_macro::TokenStream;

/// Placeholder macro; real macros are defined by the crate architecture.
#[proc_macro]
pub fn noop(input: TokenStream) -> TokenStream {
    input
}
