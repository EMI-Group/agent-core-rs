//! The RS256 JWT assertion the service-account token exchange trades for an access token.
//!
//! Google's JWT-bearer flow takes a self-signed assertion whose header names `RS256`, whose
//! claims name the service account (`iss`), the requested scope, the token endpoint (`aud`) and
//! the assertion's lifetime, and whose signature is RSASSA-PKCS1-v1_5 over SHA-256 (RFC 7515
//! `RS256`). The assertion is a compact JWS: `base64url(header).base64url(claims).base64url(
//! signature)`, with no padding and the URL-safe alphabet.
//!
//! The workspace's `sha2` has no `oid` feature, so `rsa`'s typed `SigningKey::<Sha256>` helpers
//! are unavailable. [`sha256_padding`] therefore spells the EMSA-PKCS1-v1_5 `DigestInfo` prefix
//! out — the fixed 19-byte DER prefix RFC 8017 §9.2 defines for SHA-256 — and the signature is
//! byte-for-byte the one the typed helper would produce.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use rsa::Pkcs1v15Sign;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use super::key::ServiceAccountKey;
use crate::error::{Error, Result};

/// OAuth 2.0 scope the minted access token carries.
///
/// `devstorage.read_write` is the scope the artifact transport needs: it publishes and reads
/// revisions, so read-only (`devstorage.read_only`) is not enough.
pub(super) const SCOPE: &str = "https://www.googleapis.com/auth/devstorage.read_write";

/// Lifetime of one assertion, in seconds.
///
/// Google caps an assertion's lifetime at one hour, so `exp` is always `iat + 3600`.
pub(super) const ASSERTION_LIFETIME_SECONDS: u64 = 3600;

/// The DER `DigestInfo` prefix of EMSA-PKCS1-v1_5 over SHA-256 (RFC 8017 §9.2, note 2).
pub(super) const SHA256_DIGEST_INFO_PREFIX: [u8; 19] = [
    0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05,
    0x00, 0x04, 0x20,
];

/// The RSASSA-PKCS1-v1_5 padding that reproduces `RS256`.
///
/// The typed `Pkcs1v15Sign::new::<Sha256>()` needs `Sha256: AssociatedOid`, which the workspace's
/// `sha2` does not provide, so the two public fields are filled in directly: a 32-byte hash and
/// the SHA-256 `DigestInfo` prefix. The signature is identical either way.
pub(super) fn sha256_padding() -> Pkcs1v15Sign {
    Pkcs1v15Sign {
        hash_len: Some(32),
        prefix: SHA256_DIGEST_INFO_PREFIX.to_vec().into_boxed_slice(),
    }
}

/// Builds the signed JWT assertion for `key`, issued at `issued_at` (Unix seconds).
///
/// The header is `{"alg":"RS256","typ":"JWT"}`, plus `"kid"` when the key file names a
/// `private_key_id`; the claims are `iss`, `scope`, `aud`, `iat` and `exp`, with `exp` exactly
/// [`ASSERTION_LIFETIME_SECONDS`] after `iat`.
///
/// # Errors
///
/// [`Error::External`] when a part cannot be serialized or the signature cannot be produced.
/// The message never contains the assertion, the claims or any key material.
pub(super) fn assertion(key: &ServiceAccountKey, issued_at: u64) -> Result<String> {
    let mut header = Map::new();
    header.insert("alg".to_string(), Value::String("RS256".to_string()));
    header.insert("typ".to_string(), Value::String("JWT".to_string()));
    if let Some(key_id) = key.private_key_id() {
        header.insert("kid".to_string(), Value::String(key_id.to_string()));
    }

    let mut claims = Map::new();
    claims.insert(
        "iss".to_string(),
        Value::String(key.client_email().to_string()),
    );
    claims.insert("scope".to_string(), Value::String(SCOPE.to_string()));
    claims.insert(
        "aud".to_string(),
        Value::String(key.token_uri().to_string()),
    );
    claims.insert("iat".to_string(), Value::from(issued_at));
    claims.insert(
        "exp".to_string(),
        Value::from(issued_at.saturating_add(ASSERTION_LIFETIME_SECONDS)),
    );

    let signing_input = format!(
        "{}.{}",
        encode_json(&Value::Object(header))?,
        encode_json(&Value::Object(claims))?
    );
    let signature = sign(key, signing_input.as_bytes())?;
    Ok(format!(
        "{signing_input}.{}",
        URL_SAFE_NO_PAD.encode(signature)
    ))
}

/// Serializes `value` and base64url-encodes it without padding.
fn encode_json(value: &Value) -> Result<String> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        Error::external(error).with_context("serializing the service-account JWT")
    })?;
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}

/// Signs `message` with RSASSA-PKCS1-v1_5 over SHA-256.
fn sign(key: &ServiceAccountKey, message: &[u8]) -> Result<Vec<u8>> {
    let digest = Sha256::digest(message);
    key.private_key()
        .sign(sha256_padding(), &digest)
        .map_err(|error| {
            Error::external(error).with_context("signing the service-account JWT assertion")
        })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use rsa::traits::PublicKeyParts;
    use rsa::RsaPublicKey;

    /// The same 2048-bit PKCS#8 RSA key the key-module tests use.
    const SERVICE_ACCOUNT_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDWEwu1E3LIrY/1
ukZXqTj+8wVh/tbtgPgivj104Ox3FLpvj3jhRu3QRibNMN6DM4Fi13J6IE2WRdxF
qiwyVX9lg98zpl6msySu3EYewibFHWPyW0OaKzm1GqNjzvFev/O7CqvA6KJUtgfj
rneYOGBqpl//ojbdO7d9/kq1ivXmaM6lvx+JSssHnYbvdfmH3PtSGfdtODuNSrrI
yDyAycSKmi6DIxojI3hJGXZm9oMX0Dh39sXOvTk+s66b59cDqq2TZfFoUD4Z9wtj
7b34kAZj4pGKk36OT+TDXzmypKvFbf7lHYSf+P4RN9EasunktpoRSQRlpXElNkKd
qgkZft/hAgMBAAECggEAds7TLIfCqIVK5EGhUH1mWHhfgGeCuQVesUmLqhY6pkwm
vEuP9tb6nSCR4EdC9uXa5RRVQFo1NTVIbpFnPfgdYCWz9mJbHYFG1pUBhYiTI8qP
8a2c8y8SM+S40WfEpaU6i4/8im+RyBnOfxoQ3bZMLGRx2toYQOgnihFbPS4GEv0L
8NTdj0XGNw09cy9ckIosHaZjF7MvNleeJjSR0l4Bl6HbCAS8DAxmrwXzEX6tRXdp
rYrjkqSP6qvC4+S7L4q6OpPNqRSl4urTC9ubFhtyfmo3FlMBC9w7uHLEghyfbSAF
WrHYjTxgRZnifeb9RfPfxButnQ4Y2lExmBysuaQzUQKBgQDmQ4tvP6wKHFwxljzG
ioyhJlWNsekJ306IVvLPd7fGfosJOIyVe/s1CJJa8hbNETZuW4TqyNVwBBKJYFHd
qBvJbOO5uxCb3kd9SoVR5GKM/XJm1344G1bBNil2fize+meza3YIs9bFP1DdJFeL
xxtIhfAAsehpsiaXLdO49/Cs/wKBgQDuAEc9DeIO5f6MuTijS2uvR8MrRcCap0PC
hYWjfeibYrg3jMAZuGmIEDY8UMwDIyTC5adsgnvbBwa9kuu1QJoOQkgpynSncdHq
rWRui1Vreqzwjwgo+vlPjP7UCNmmmsX5h4Lq/piKZRoNDHJuYbrsP4GLHYqXu708
nYriQn0THwKBgD+/s3q5RggGkf+oeGr/MK/szQLNnpb9G+DjMZX+rwJlRljnLwSz
rnigRuSm4Wkkus0SyVm3AGBiwD8IS6HySL8bRnDFOXPtth0Z5I8ni6E9Cc8vGeKu
rvYO6KvFgua4rL0foy4FnBA/10NiLFqmZQdPDYMjvhSFHen85gDvo9JBAoGAErlq
4V7KPinSEPgxgIdafXvD2w+ty30FU7R1gxtQxYjKvXQRWPStOMstuSkSI90L81lm
Bp9tZxnyac+5Q5VFiYGlc8Q90T6fKTti/19u5PnNyU83cSHto1iQY9YeaYEbujnz
DtHg5mJ13jjwd8awkg2c/3jfgnh+ttoeJxbcCDkCgYEA1wWLpUIeRchZHOqbYFoM
QEDRqvnHYqrd0cp9hTp+uyoCz9e8C8n54wKzgWEtkxsAfO3Pami4T+z+NiYeFpWn
+1SJIbJWbPf2elxD60UjEa8OeelSGiu5X5KlfPvyuVgX4qs9vUYQQiTHDZbzmDlV
I1Ru6xykeZLm6JI5YG+U0zg=
-----END PRIVATE KEY-----
";

    /// The token endpoint every test key points at.
    const TOKEN_URI: &str = "https://oauth2.googleapis.com/token";

    /// Builds a key from a complete document, optionally with a `private_key_id`.
    fn key(with_key_id: bool) -> ServiceAccountKey {
        let mut document = serde_json::json!({
            "type": "service_account",
            "private_key": SERVICE_ACCOUNT_PEM,
            "client_email": "symphony@demo-project.iam.gserviceaccount.com",
            "token_uri": TOKEN_URI,
        });
        if with_key_id {
            document["private_key_id"] = serde_json::Value::String("0123456789abcdef".to_string());
        }
        ServiceAccountKey::from_json(&document, "<test>").expect("parse the key")
    }

    /// Splits an assertion into its three decoded-or-raw segments.
    fn segments(assertion: &str) -> (String, String, String, String) {
        let parts: Vec<&str> = assertion.split('.').collect();
        assert_eq!(parts.len(), 3, "{assertion}");
        for part in &parts {
            assert!(!part.is_empty(), "{assertion}");
            assert!(!part.contains('='), "{assertion}");
            assert!(!part.contains('+'), "{assertion}");
            assert!(!part.contains('/'), "{assertion}");
        }
        (
            parts[0].to_string(),
            parts[1].to_string(),
            parts[2].to_string(),
            format!("{}.{}", parts[0], parts[1]),
        )
    }

    /// Decodes one base64url segment as JSON.
    fn decode_json(segment: &str) -> serde_json::Value {
        let bytes = URL_SAFE_NO_PAD.decode(segment).expect("decode a segment");
        serde_json::from_slice(&bytes).expect("decode a segment as JSON")
    }

    #[test]
    fn the_assertion_carries_the_documented_header_and_claims() {
        let key = key(true);
        let assertion = assertion(&key, 1_700_000_000).expect("build the assertion");
        let (header, claims, ..) = segments(&assertion);

        let header = decode_json(&header);
        assert_eq!(header["alg"], "RS256");
        assert_eq!(header["typ"], "JWT");
        assert_eq!(header["kid"], "0123456789abcdef");

        let claims = decode_json(&claims);
        assert_eq!(
            claims["iss"],
            "symphony@demo-project.iam.gserviceaccount.com"
        );
        assert_eq!(claims["scope"], SCOPE);
        assert_eq!(claims["aud"], TOKEN_URI);
        assert_eq!(claims["iat"], 1_700_000_000);
        assert_eq!(claims["exp"], 1_700_000_000 + 3600);
        assert_eq!(
            claims["exp"].as_u64().expect("a number") - 3600,
            1_700_000_000
        );
        assert_eq!(
            SCOPE,
            "https://www.googleapis.com/auth/devstorage.read_write"
        );
        assert_eq!(ASSERTION_LIFETIME_SECONDS, 3600);
    }

    #[test]
    fn the_kid_is_absent_when_the_key_names_no_private_key_id() {
        let key = key(false);
        let assertion = assertion(&key, 1).expect("build the assertion");
        let (header, ..) = segments(&assertion);
        let header = decode_json(&header);
        assert!(header.get("kid").is_none(), "{header}");
    }

    #[test]
    fn the_signature_verifies_with_the_public_key_as_rs256() {
        let key = key(true);
        let assertion = assertion(&key, 1_700_000_000).expect("build the assertion");
        let (_, _, signature, signing_input) = segments(&assertion);
        let signature = URL_SAFE_NO_PAD
            .decode(signature)
            .expect("decode the signature");
        assert_eq!(signature.len(), 256);
        assert_eq!(key.private_key().to_public_key().size(), 256);

        let public: &RsaPublicKey = &key.private_key().to_public_key();
        let digest = Sha256::digest(signing_input.as_bytes());
        public
            .verify(sha256_padding(), &digest, &signature)
            .expect("the signature must verify");

        // One flipped byte of the signing input must not verify.
        let mut tampered = signing_input.into_bytes();
        let last = tampered.len() - 1;
        tampered[last] ^= 0x01;
        let digest = Sha256::digest(&tampered);
        assert!(public
            .verify(sha256_padding(), &digest, &signature)
            .is_err());
    }

    #[test]
    fn the_padding_prefix_is_the_sha256_digest_info() {
        assert_eq!(
            SHA256_DIGEST_INFO_PREFIX,
            [
                0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02,
                0x01, 0x05, 0x00, 0x04, 0x20,
            ]
        );
        let padding = sha256_padding();
        assert_eq!(padding.hash_len, Some(32));
        assert_eq!(&*padding.prefix, &SHA256_DIGEST_INFO_PREFIX);
    }

    #[test]
    fn assertions_are_deterministic_for_one_instant() {
        let key = key(true);
        let first = assertion(&key, 1_700_000_000).expect("build");
        let second = assertion(&key, 1_700_000_000).expect("build again");
        assert_eq!(first, second);
        assert_ne!(first, assertion(&key, 1_700_000_001).expect("build"));
    }
}
