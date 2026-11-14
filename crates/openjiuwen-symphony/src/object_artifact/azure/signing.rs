//! Azure Blob Shared Key signing.
//!
//! The workspace has no HMAC crate, so [`SigningKey`] decodes the base64 account key once and
//! signs the documented [`string_to_sign`] layout with the hand-rolled
//! [`hmac_sha256`](super::super::http::hmac_sha256) the S3 signer uses.

use std::collections::BTreeMap;
use std::fmt;

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use openjiuwen_common::security::SecretString;

use super::Verb;
use crate::error::{validation, Result};

/// The authorization scheme name prefixed to every signature.
pub(super) const SCHEME: &str = "SharedKey";

/// Decoded Shared Key material.
#[derive(Clone)]
pub(super) struct SigningKey {
    /// Storage account the key belongs to.
    pub(super) account: String,
    /// Decoded key bytes.
    pub(super) key: Vec<u8>,
}

impl fmt::Debug for SigningKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SigningKey")
            .field("account", &self.account)
            .field("key", &"<redacted>")
            .finish()
    }
}

/// Builds the Shared Key string to sign for one request.
///
/// The layout is the documented one: the verb, one line per Content-* / Date / conditional
/// field (all empty here), `Content-Length` (empty for `0` or no body), the canonicalized
/// `x-ms-*` headers sorted by name, and the canonicalized resource `/{account}{path}` followed
/// by one `\nname:value` line per query parameter sorted by name. Each canonicalized header
/// line ends with `\n`, so the resource follows the last one immediately.
pub(super) fn string_to_sign(
    verb: Verb,
    content_length: Option<u64>,
    headers: &BTreeMap<String, String>,
    query: &[(String, String)],
    path: &str,
    account: &str,
) -> String {
    let mut canonical_headers: Vec<(String, String)> = headers
        .iter()
        .filter(|(name, _)| name.to_ascii_lowercase().starts_with("x-ms-"))
        .map(|(name, value)| (name.to_ascii_lowercase(), canonical_value(value)))
        .collect();
    canonical_headers.sort();
    let mut canonicalized_headers = String::new();
    for (name, value) in &canonical_headers {
        canonicalized_headers.push_str(name);
        canonicalized_headers.push(':');
        canonicalized_headers.push_str(value);
        canonicalized_headers.push('\n');
    }

    let mut canonicalized_resource = format!("/{account}{path}");
    let mut params: Vec<(String, String)> = query
        .iter()
        .map(|(name, value)| (name.to_ascii_lowercase(), value.clone()))
        .collect();
    params.sort();
    for (name, value) in &params {
        canonicalized_resource.push('\n');
        canonicalized_resource.push_str(name);
        canonicalized_resource.push(':');
        canonicalized_resource.push_str(value);
    }

    let length = content_length
        .filter(|length| *length > 0)
        .map(|length| length.to_string())
        .unwrap_or_default();
    let mut string = String::new();
    for line in [
        verb.as_str(),
        "", // Content-Encoding
        "", // Content-Language
        length.as_str(),
        "", // Content-MD5
        "", // Content-Type
        "", // Date (the request carries `x-ms-date` instead)
        "", // If-Modified-Since
        "", // If-Match
        "", // If-None-Match
        "", // If-Unmodified-Since
        "", // Range
    ] {
        string.push_str(line);
        string.push('\n');
    }
    string.push_str(&canonicalized_headers);
    string.push_str(&canonicalized_resource);
    string
}

/// Normalizes the value of one canonicalized header: trimmed, with no line breaks.
fn canonical_value(value: &str) -> String {
    value.replace(['\r', '\n'], "").trim().to_string()
}

impl SigningKey {
    /// Decodes the base64 key of a [`AzureBlobAuth::SharedKey`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming `account` for a blank account and `key` for a key
    /// that is blank or not base64.
    pub(super) fn new(account: &str, key: &SecretString) -> Result<Self> {
        let account = account.trim();
        if account.is_empty() {
            return Err(validation(
                "account",
                "must not be blank when authenticating with a shared key",
            ));
        }
        let encoded = key.expose_secret().trim();
        if encoded.is_empty() {
            return Err(validation("key", "must not be blank"));
        }
        let decoded = STANDARD.decode(encoded).map_err(|error| {
            validation(
                "key",
                format!("an Azure storage account key is base64: {error}"),
            )
        })?;
        Ok(Self {
            account: account.to_string(),
            key: decoded,
        })
    }
}
