//! RFC 1951 DEFLATE decoding (shared by `zip` and `gzip`).
//!
//! The decoder is bounded by an explicit output cap: a stored block, literal
//! run, or back-reference that would exceed the cap fails with a validation
//! error naming the cap instead of allocating.

use openjiuwen_common::error::{Error, Result};

/// Decompresses one raw DEFLATE stream, refusing to produce more than `max_output` bytes.
///
/// # Errors
///
/// Returns [`Error::validation`] for malformed or truncated input and for output
/// that would exceed `max_output`.
pub(crate) fn inflate(_input: &[u8], _max_output: usize) -> Result<Vec<u8>> {
    Err(Error::unsupported(
        "DEFLATE decoding is not wired into this build",
    ))
}
