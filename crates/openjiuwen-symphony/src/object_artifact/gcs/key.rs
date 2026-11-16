//! Parsing of a Google Cloud Storage service-account JSON key file.
//!
//! A GCS service-account key file is the JSON document `gcloud iam service-accounts keys
//! create` (or the console's *Create key → JSON*) writes. The store only needs five fields:
//! the credential `type`, the `client_email` that signs the token request, the `private_key`
//! the assertion is signed with, the `token_uri` the assertion is posted to, and — optionally —
//! the `project_id` (a bucket operation needs a project) and `private_key_id` (the JWT header's
//! `kid`).
//!
//! The whole document is accepted with its extra fields (`private_key_id`, `universe_domain`,
//! `auth_uri`, …): Google adds fields over time, so an unknown key is tolerated rather than
//! rejected.
//!
//! [`ServiceAccountKey`] holds a live RSA private key, so its [`Debug`] implementation prints
//! only the public fields and a `"<redacted>"` marker — the derived implementation of
//! [`RsaPrivateKey`] would print the key material.

use std::fmt;
use std::path::Path;

use rsa::pkcs8::DecodePrivateKey;
use rsa::RsaPrivateKey;
use serde_json::{Map, Value};

use crate::error::{validation, Error, Result};

/// A parsed service-account key file.
///
/// The private key is kept as an [`RsaPrivateKey`] so the assertion is signed without parsing
/// the PEM again on every token refresh.
pub(super) struct ServiceAccountKey {
    /// Service-account address the assertion is issued for (`iss`).
    client_email: String,

    /// Optional id of the key pair, sent as the JWT header's `kid`.
    private_key_id: Option<String>,

    /// OAuth 2.0 token endpoint the assertion is posted to (`aud`).
    token_uri: String,

    /// Optional Google Cloud project the service account belongs to.
    project_id: Option<String>,

    /// Parsed PKCS#8 RSA private key the assertion is signed with.
    private_key: RsaPrivateKey,
}

impl ServiceAccountKey {
    /// Reads and parses the key file at `path`.
    ///
    /// The read is blocking, because the store's constructors are synchronous: a key file is
    /// read once when the transport is built, never on a request path.
    ///
    /// # Errors
    ///
    /// [`Error::External`] naming `path` when the file cannot be read, and
    /// [`Error::Config`] naming `path` when it is not valid JSON or not a service-account key.
    pub(super) fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|error| {
            Error::external(error).with_context(format!(
                "reading the service-account key file {}",
                path.display()
            ))
        })?;
        let document: Value = serde_json::from_str(&text).map_err(|error| {
            Error::config(format!(
                "service-account key file {} is not valid JSON ({error})",
                path.display()
            ))
        })?;
        Self::from_json(&document, &path.display().to_string())
    }

    /// Parses a decoded key document, naming `source` in every error.
    ///
    /// `source` is the file path or another human-readable origin; it is only ever used in
    /// error messages, never echoed with the document's contents.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] when the document is not an object, a required field is missing, or
    /// `private_key`/`token_uri` cannot be used; [`Error::Validation`] naming the offending
    /// field when a required field has the wrong type or a blank value.
    pub(super) fn from_json(value: &Value, source: &str) -> Result<Self> {
        let Some(document) = value.as_object() else {
            return Err(Error::config(format!(
                "service-account key file {source} must contain a JSON object"
            )));
        };

        let key_type = required_string(document, "type", source)?;
        if key_type != "service_account" {
            return Err(validation(
                "type",
                format!("expected `service_account`, got `{key_type}` in {source}"),
            ));
        }

        let client_email = required_string(document, "client_email", source)?;
        let token_uri = required_string(document, "token_uri", source)?;
        validate_token_uri(&token_uri, source)?;

        let pem = required_string(document, "private_key", source)?;
        let private_key = RsaPrivateKey::from_pkcs8_pem(&pem).map_err(|error| {
            Error::config(format!(
                "service-account key file {source}: `private_key` is not a PKCS#8 RSA private \
                 key PEM ({error})"
            ))
        })?;

        Ok(Self {
            client_email,
            private_key_id: optional_string(document, "private_key_id"),
            token_uri,
            project_id: optional_string(document, "project_id"),
            private_key,
        })
    }

    /// The service-account address the assertion is issued for.
    pub(super) fn client_email(&self) -> &str {
        &self.client_email
    }

    /// The OAuth 2.0 token endpoint the assertion is posted to.
    pub(super) fn token_uri(&self) -> &str {
        &self.token_uri
    }

    /// The Google Cloud project the service account belongs to, when the file names one.
    pub(super) fn project_id(&self) -> Option<&str> {
        self.project_id.as_deref()
    }

    /// The id of the key pair, sent as the JWT header's `kid`, when the file names one.
    pub(super) fn private_key_id(&self) -> Option<&str> {
        self.private_key_id.as_deref()
    }

    /// The parsed private key the assertion is signed with.
    pub(super) fn private_key(&self) -> &RsaPrivateKey {
        &self.private_key
    }
}

impl fmt::Debug for ServiceAccountKey {
    /// Renders the public fields only: [`RsaPrivateKey`]'s own `Debug` would print the key
    /// material, so the private key is always reported as `"<redacted>"`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ServiceAccountKey")
            .field("client_email", &self.client_email)
            .field("token_uri", &self.token_uri)
            .field("private_key_id", &self.private_key_id)
            .field("project_id", &self.project_id)
            .field("private_key", &"<redacted>")
            .finish()
    }
}

/// Reads a required, non-blank string field.
fn required_string(document: &Map<String, Value>, field: &str, source: &str) -> Result<String> {
    match document.get(field) {
        None | Some(Value::Null) => Err(Error::config(format!(
            "service-account key file {source} is missing `{field}`"
        ))),
        Some(Value::String(text)) if text.trim().is_empty() => {
            Err(validation(field, format!("must not be blank in {source}")))
        }
        Some(Value::String(text)) => Ok(text.trim().to_string()),
        Some(_) => Err(validation(field, format!("must be a string in {source}"))),
    }
}

/// Reads an optional string field; an absent, blank or non-string value is `None`.
fn optional_string(document: &Map<String, Value>, field: &str) -> Option<String> {
    document
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// Rejects a `token_uri` that is not an absolute HTTP(S) URL.
///
/// The endpoint is hand-checked rather than parsed, matching the store's endpoint validation:
/// this crate carries no URL crate.
fn validate_token_uri(uri: &str, source: &str) -> Result<()> {
    let valid = uri.split_once("://").is_some_and(|(scheme, rest)| {
        matches!(scheme, "http" | "https")
            && !rest
                .split(['/', '?', '#'])
                .next()
                .unwrap_or_default()
                .is_empty()
    });
    if valid {
        Ok(())
    } else {
        Err(Error::config(format!(
            "service-account key file {source}: `token_uri` must be an absolute http(s) URL, \
             got {uri:?}"
        )))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use rsa::traits::PublicKeyParts;

    /// A 2048-bit PKCS#8 RSA key used by every unit test in this module.
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

    /// Builds a complete key document for `token_uri`.
    fn key_document(token_uri: &str) -> Value {
        serde_json::json!({
            "type": "service_account",
            "project_id": "demo-project",
            "private_key_id": "0123456789abcdef",
            "private_key": SERVICE_ACCOUNT_PEM,
            "client_email": "symphony@demo-project.iam.gserviceaccount.com",
            "client_id": "1234567890",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": token_uri,
            "universe_domain": "googleapis.com",
        })
    }

    /// The token endpoint every test document points at.
    const TOKEN_URI: &str = "https://oauth2.googleapis.com/token";

    #[test]
    fn a_minimal_key_document_parses_and_tolerates_extra_fields() {
        let key = ServiceAccountKey::from_json(&key_document(TOKEN_URI), "<test>")
            .expect("parse a service-account key");
        assert_eq!(
            key.client_email(),
            "symphony@demo-project.iam.gserviceaccount.com"
        );
        assert_eq!(key.token_uri(), TOKEN_URI);
        assert_eq!(key.project_id(), Some("demo-project"));
        assert_eq!(key.private_key_id(), Some("0123456789abcdef"));
        assert_eq!(key.private_key().size(), 256);
    }

    #[test]
    fn optional_fields_are_optional_and_blank_values_become_none() {
        let document = serde_json::json!({
            "type": "service_account",
            "client_email": " symphony@example.com ",
            "private_key": SERVICE_ACCOUNT_PEM,
            "token_uri": TOKEN_URI,
            "project_id": "  ",
            "private_key_id": "",
        });
        let key = ServiceAccountKey::from_json(&document, "<test>").expect("parse");
        assert_eq!(key.client_email(), "symphony@example.com");
        assert!(key.project_id().is_none());
        assert!(key.private_key_id().is_none());

        let document = serde_json::json!({
            "type": "service_account",
            "client_email": "symphony@example.com",
            "private_key": SERVICE_ACCOUNT_PEM,
            "token_uri": TOKEN_URI,
        });
        let key = ServiceAccountKey::from_json(&document, "<test>").expect("parse");
        assert!(key.project_id().is_none());
        assert!(key.private_key_id().is_none());
    }

    #[test]
    fn every_required_field_is_named_when_missing() {
        for field in ["type", "client_email", "private_key", "token_uri"] {
            let mut document = key_document(TOKEN_URI);
            document.as_object_mut().expect("an object").remove(field);
            let error = ServiceAccountKey::from_json(&document, "/tmp/sa.json")
                .expect_err("a missing field must be rejected");
            assert_eq!(error.kind(), crate::error::ErrorKind::Config, "{field}");
            assert!(error.message().contains(field), "{field}: {error}");
            assert!(error.message().contains("/tmp/sa.json"), "{field}: {error}");
        }
    }

    #[test]
    fn the_credential_type_must_be_a_service_account() {
        let mut document = key_document(TOKEN_URI);
        document["type"] = Value::String("authorized_user".to_string());
        let error = ServiceAccountKey::from_json(&document, "<test>").expect_err("wrong type");
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("service_account"), "{error}");
        assert!(error.message().contains("authorized_user"), "{error}");
    }

    #[test]
    fn blank_and_mistyped_required_fields_are_validation_errors() {
        for field in ["client_email", "private_key", "token_uri"] {
            let mut document = key_document(TOKEN_URI);
            document[field] = Value::String("   ".to_string());
            let error = ServiceAccountKey::from_json(&document, "<test>").expect_err("blank field");
            assert_eq!(error.kind(), crate::error::ErrorKind::Validation, "{field}");
            assert!(error.message().contains(field), "{field}: {error}");

            let mut document = key_document(TOKEN_URI);
            document[field] = Value::Number(7.into());
            let error =
                ServiceAccountKey::from_json(&document, "<test>").expect_err("mistyped field");
            assert_eq!(error.kind(), crate::error::ErrorKind::Validation, "{field}");
            assert!(error.message().contains(field), "{field}: {error}");
        }
    }

    #[test]
    fn the_token_uri_must_be_an_absolute_http_url() {
        for token_uri in ["ftp://example.com/token", "/token", "token"] {
            let error = ServiceAccountKey::from_json(&key_document(token_uri), "<test>")
                .expect_err("a non-http token_uri must be rejected");
            assert_eq!(
                error.kind(),
                crate::error::ErrorKind::Config,
                "{token_uri:?}"
            );
            assert!(error.message().contains("token_uri"), "{error}");
        }
        // A blank value is a validation failure, like every other blank required field.
        let error = ServiceAccountKey::from_json(&key_document(""), "<test>")
            .expect_err("a blank token_uri must be rejected");
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("token_uri"), "{error}");

        assert!(ServiceAccountKey::from_json(
            &key_document("http://127.0.0.1:8080/token"),
            "<test>"
        )
        .is_ok());
    }

    #[test]
    fn a_malformed_private_key_pem_is_rejected_without_echoing_it() {
        let mut document = key_document(TOKEN_URI);
        document["private_key"] = Value::String("not a pem".to_string());
        let error = ServiceAccountKey::from_json(&document, "<test>").expect_err("bad pem");
        assert_eq!(error.kind(), crate::error::ErrorKind::Config);
        assert!(error.message().contains("private_key"), "{error}");
        assert!(!error.message().contains("MII"), "{error}");
    }

    #[test]
    fn a_non_rsa_pkcs8_key_is_rejected() {
        // A PKCS#8 document for an EC key: a v1 wrapper, the `id-ecPublicKey` algorithm with a
        // named curve, and an opaque private-key octet string. `rsa` refuses it on the
        // algorithm identifier, which is exactly the failure reported to the caller.
        const EC_PKCS8_DER: &[u8] = &[
            0x30, 0x1e, 0x02, 0x01, 0x00, 0x30, 0x13, 0x06, 0x07, 0x2a, 0x86, 0x48, 0xce, 0x3d,
            0x02, 0x01, 0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07, 0x04, 0x04,
            0x04, 0x01, 0x02, 0x03,
        ];
        use base64::Engine as _;
        let pem = format!(
            "-----BEGIN PRIVATE KEY-----\n{}\n-----END PRIVATE KEY-----\n",
            base64::engine::general_purpose::STANDARD.encode(EC_PKCS8_DER)
        );
        let mut document = key_document(TOKEN_URI);
        document["private_key"] = Value::String(pem);
        let error = ServiceAccountKey::from_json(&document, "<test>").expect_err("an EC key");
        assert_eq!(error.kind(), crate::error::ErrorKind::Config);
        assert!(error.message().contains("private_key"), "{error}");
    }

    #[test]
    fn a_key_file_round_trips_through_from_file() {
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("service-account.json");
        std::fs::write(
            &path,
            serde_json::to_vec(&key_document(TOKEN_URI)).expect("serialize"),
        )
        .expect("write the key file");

        let key = ServiceAccountKey::from_file(&path).expect("read the key file");
        assert_eq!(key.token_uri(), TOKEN_URI);
        assert_eq!(key.project_id(), Some("demo-project"));
    }

    #[test]
    fn an_unreadable_key_file_names_the_path() {
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("absent.json");
        let error = ServiceAccountKey::from_file(&path).expect_err("a missing file");
        assert_eq!(error.kind(), crate::error::ErrorKind::External);
        assert!(
            error.message().contains("absent.json"),
            "{}",
            error.message()
        );
    }

    #[test]
    fn a_malformed_key_file_names_the_path() {
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("broken.json");
        std::fs::write(&path, "{not json").expect("write");
        let error = ServiceAccountKey::from_file(&path).expect_err("malformed JSON");
        assert_eq!(error.kind(), crate::error::ErrorKind::Config);
        assert!(error.message().contains("broken.json"), "{error}");
        assert!(!error.message().contains("not json"), "{error}");
    }

    #[test]
    fn debug_never_prints_the_private_key() {
        let key = ServiceAccountKey::from_json(&key_document(TOKEN_URI), "<test>").expect("parse");
        let rendered = format!("{key:?}");
        assert!(rendered.contains("symphony@demo-project.iam.gserviceaccount.com"));
        assert!(rendered.contains(TOKEN_URI));
        assert!(rendered.contains("0123456789abcdef"));
        assert!(rendered.contains("<redacted>"), "{rendered}");
        assert!(!rendered.contains("MII"), "{rendered}");
        assert!(!rendered.contains("BEGIN PRIVATE KEY"), "{rendered}");
    }
}
