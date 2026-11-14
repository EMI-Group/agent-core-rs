//! Truncated SHA-256 digests for stable harness-side identifiers.
//!
//! Python derives `PersonalContext` source ids (`src_<32 hex>`), source-link
//! identities and KV-cache sub-session scopes from SHA-256, so the ids stay
//! wire-compatible only when the digest matches. The digest itself is
//! [`openjiuwen_sysop::util::sha256_hex`] — the workspace's `sha2`-based
//! implementation, already used by the persistent-agent journal — so this
//! module only keeps the truncation wrapper every call site shares.
//!
//! This is an identifier digest, not a security primitive: it makes ids stable
//! and collision-resistant across runs, and it is never used to authenticate,
//! sign or encrypt anything. Secrets never enter it.

/// Returns the first `hex_chars` characters of the hex SHA-256 digest of `input`.
///
/// `hex_chars` is clamped to the 64 characters a digest has, so an oversized
/// request degrades to the full digest instead of panicking.
pub(crate) fn sha256_prefix(input: impl AsRef<[u8]>, hex_chars: usize) -> String {
    let digest = openjiuwen_sysop::util::sha256_hex(input.as_ref());
    let end = hex_chars.min(digest.len());
    digest[..end].to_string()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_sysop::util::sha256_hex;

    use super::*;

    /// FIPS 180-4 / NIST examples.
    const EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    #[test]
    fn the_underlying_digest_matches_the_published_vectors() {
        assert_eq!(sha256_hex(b""), EMPTY);
        assert_eq!(sha256_hex(b"abc"), ABC);
    }

    #[test]
    fn prefix_truncates_and_clamps() {
        assert_eq!(sha256_prefix("abc", 12), &ABC[..12]);
        assert_eq!(sha256_prefix("abc", 12).len(), 12);
        assert_eq!(sha256_prefix("abc", 0), "");
        assert_eq!(sha256_prefix("abc", 1000), ABC);
    }
}
