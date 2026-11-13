//! `ACS3-HMAC-SHA256` signing helpers for the Aliyun Green adapter.
//!
//! The adapter authenticates an access-key request with a signature over the
//! `POST` request line, six canonical headers, and the SHA-256 of the request
//! body: [`canonical_request`] builds that payload, [`sign_authorization`]
//! hashes it and keys the RFC 2104 `HMAC` with the access key secret, and
//! [`sha256_hex`] also supplies the `x-acs-content-sha256` header value.
//! Every digest is lowercase hex.
//!
//! The helpers are pure and deterministic: the same `(host, api_version, date,
//! nonce, body_sha)` inputs always produce the same `Authorization` value.

use sha2::{Digest, Sha256};

use super::ACTION;

/// HMAC block size in bytes (SHA-256 uses 64-byte blocks).
const HMAC_BLOCK_SIZE: usize = 64;
/// Signature algorithm name, used for the string to sign and the header.
const SIGNATURE_ALGORITHM: &str = "ACS3-HMAC-SHA256";
/// Sorted, semicolon-joined names of the headers the signature covers.
pub(super) const SIGNED_HEADER_NAMES: &str =
    "host;x-acs-action;x-acs-content-sha256;x-acs-date;x-acs-signature-nonce;x-acs-version";

/// Composes the RFC 2104 HMAC-SHA256 of `data` under `key`.
fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut block = [0_u8; HMAC_BLOCK_SIZE];
    if key.len() > HMAC_BLOCK_SIZE {
        block[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(block.map(|byte| byte ^ 0x36));
    inner.update(data);
    let inner = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(block.map(|byte| byte ^ 0x5c));
    outer.update(inner);
    let mut digest = [0_u8; 32];
    digest.copy_from_slice(&outer.finalize());
    digest
}

/// Lowercase-hex HMAC-SHA256 digest of `data` under `key`.
fn hmac_sha256_hex(key: &[u8], data: &[u8]) -> String {
    hex_encode(&hmac_sha256(key, data))
}

/// Lowercase-hex SHA-256 digest of `bytes`.
pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    hex_encode(&Sha256::digest(bytes))
}

/// Encodes `bytes` as a lowercase hexadecimal string.
fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

/// Builds the `ACS3-HMAC-SHA256` canonical request.
fn canonical_request(
    host: &str,
    api_version: &str,
    date: &str,
    nonce: &str,
    body_sha: &str,
) -> String {
    let headers = format!(
        "host:{host}\n\
         x-acs-action:{ACTION}\n\
         x-acs-content-sha256:{body_sha}\n\
         x-acs-date:{date}\n\
         x-acs-signature-nonce:{nonce}\n\
         x-acs-version:{api_version}\n"
    );
    format!("POST\n/\n{headers}\n{SIGNED_HEADER_NAMES}\n{body_sha}")
}

/// Builds the `Authorization` value for an `Aksk` credential.
///
/// The signature is deterministic in `(host, api_version, date, nonce,
/// body_sha)`, so the same request inputs always produce the same header.
pub(super) fn sign_authorization(
    access_key_id: &str,
    access_key_secret: &str,
    host: &str,
    api_version: &str,
    date: &str,
    nonce: &str,
    body_sha: &str,
) -> String {
    let canonical = canonical_request(host, api_version, date, nonce, body_sha);
    let string_to_sign = format!(
        "{SIGNATURE_ALGORITHM}\n{}",
        sha256_hex(canonical.as_bytes())
    );
    let signature = hmac_sha256_hex(access_key_secret.as_bytes(), string_to_sign.as_bytes());
    format!(
        "{SIGNATURE_ALGORITHM} Credential={access_key_id},\
         SignedHeaders={SIGNED_HEADER_NAMES},Signature={signature}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const ACCESS_KEY_ID: &str = "test-access-key-id";
    const ACCESS_KEY_SECRET: &str = "test-access-key-secret";

    #[test]
    fn digest_vectors_match_the_rfcs() {
        let vectors: [(&[u8], &[u8], &str); 3] = [
            (
                &[0x0b_u8; 20],
                b"Hi There",
                "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7",
            ),
            (
                b"Jefe",
                b"what do ya want for nothing?",
                "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
            ),
            // A longer-than-block-size key exercises the hashed-key path.
            (
                &[0xaa_u8; 131],
                b"Test Using Larger Than Block-Size Key - Hash Key First",
                "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54",
            ),
        ];
        for (key, data, expected) in vectors {
            assert_eq!(hmac_sha256_hex(key, data), expected);
        }
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(hex_encode(&[0x00, 0x0f, 0xff]), "000fff");
    }

    #[test]
    fn canonical_request_and_signature_are_deterministic() {
        let canonical = canonical_request(
            "green.example",
            "2022-03-02",
            "2026-01-01T00:00:00Z",
            "nonce-1",
            "abc123",
        );
        assert_eq!(
            canonical,
            "POST\n/\nhost:green.example\nx-acs-action:TextModeration\n\
             x-acs-content-sha256:abc123\nx-acs-date:2026-01-01T00:00:00Z\n\
             x-acs-signature-nonce:nonce-1\nx-acs-version:2022-03-02\n\n\
             host;x-acs-action;x-acs-content-sha256;x-acs-date;x-acs-signature-nonce;x-acs-version\nabc123"
        );
        let sign = |host: &str| {
            sign_authorization(
                ACCESS_KEY_ID,
                ACCESS_KEY_SECRET,
                host,
                "2022-03-02",
                "2026-01-01T00:00:00Z",
                "nonce-1",
                "abc123",
            )
        };
        let first = sign("green.example");
        assert_eq!(first, sign("green.example"));
        assert_ne!(first, sign("other.example"));
        let string_to_sign = format!("ACS3-HMAC-SHA256\n{}", sha256_hex(canonical.as_bytes()));
        let expected = format!(
            "ACS3-HMAC-SHA256 Credential={ACCESS_KEY_ID},SignedHeaders={SIGNED_HEADER_NAMES},Signature={}",
            hmac_sha256_hex(ACCESS_KEY_SECRET.as_bytes(), string_to_sign.as_bytes())
        );
        assert_eq!(first, expected);
        assert!(!first.contains(ACCESS_KEY_SECRET));
    }
}
