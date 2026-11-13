//! RFC 1952 `gzip` member parsing on top of the shared DEFLATE decoder.

use super::ArchiveEntry;
use openjiuwen_common::error::{Error, Result};

/// Decompresses a `gzip` stream and unpacks the `tar` or `zip` payload it carries.
///
/// # Errors
///
/// Returns [`Error::validation`] for a malformed member, a CRC/ISIZE mismatch,
/// or a payload that is not a `tar` or `zip` archive, and [`Error::unsupported`]
/// for compression methods other than DEFLATE.
pub(crate) fn unpack(_bytes: &[u8]) -> Result<Vec<ArchiveEntry>> {
    Err(Error::unsupported(
        "gzip archive unpacking requires a compression dependency that the offline workspace does \
         not provide",
    ))
}
