//! Hashing helpers ported from Python `openjiuwen.core.common.utils.hash_util`.

use sha2::{Digest, Sha256};

/// Builds the cache key identifying an LLM endpoint.
///
/// This is a port of Python `utils.hash_util.generate_key`: the three inputs are
/// sorted lexicographically, concatenated without separators, and hashed with
/// SHA-256. The result is a 64-character lowercase hex digest.
///
/// Because the inputs are sorted, the argument order does not affect the result.
///
/// ```
/// use openjiuwen_common::utils::hash::generate_key;
///
/// assert_eq!(generate_key("a", "b", "c"), generate_key("c", "b", "a"));
/// ```
#[must_use]
pub fn generate_key(api_key: &str, api_base: &str, model_provider: &str) -> String {
    let mut parts = [api_key, api_base, model_provider];
    parts.sort_unstable();

    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // port of tests/unit_tests/core/common/utils/test_hash_util.py

    /// `sha256("abc")`, i.e. the sorted concatenation of `"a"`, `"b"`, `"c"`.
    const SHA256_ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    /// `sha256("")`, i.e. the sorted concatenation of three empty strings.
    const SHA256_EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    #[test]
    fn matches_python_sorted_concatenation() {
        assert_eq!(generate_key("a", "b", "c"), SHA256_ABC);
        assert_eq!(generate_key("c", "a", "b"), SHA256_ABC);
        assert_eq!(generate_key("", "", ""), SHA256_EMPTY);
    }

    #[test]
    fn is_a_lowercase_hex_digest() {
        let key = generate_key("sk-test", "https://api.example.com/v1", "openai");
        assert_eq!(key.len(), 64);
        assert!(key
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }

    #[test]
    fn differs_when_inputs_differ() {
        let base = generate_key("sk-test", "https://api.example.com/v1", "openai");
        assert_ne!(
            base,
            generate_key("sk-other", "https://api.example.com/v1", "openai")
        );
        assert_ne!(
            base,
            generate_key("sk-test", "https://api.example.com/v2", "openai")
        );
        assert_ne!(
            base,
            generate_key("sk-test", "https://api.example.com/v1", "anthropic")
        );
    }
}
