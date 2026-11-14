//! Endpoint resolution, path encoding and name validation for the Azure Blob transport.
//!
//! The Blob API addresses a container as one path segment and a blob as a `/`-separated path,
//! so the two are encoded differently: [`encode_blob_name`] keeps the separators and encodes
//! every other reserved byte, while [`validate_container`] rejects a container name that cannot
//! be a segment at all. Everything here is pure, so it is shared by the store and by the SAS /
//! endpoint resolution in the parent module.

use super::{AzureBlobConfig, AZURITE_ENDPOINT, DEFAULT_ENDPOINT_SUFFIX};
use crate::error::{validation, Error, Result};

/// Uppercase hexadecimal digits of a percent escape.
const HEX_DIGITS: &[u8; 16] = b"0123456789ABCDEF";

/// Percent-encodes a blob name, keeping `/` as the segment separator.
///
/// A Blob name is a multi-segment path, so unlike a GCS object name it keeps its separators;
/// every other byte outside the RFC 3986 unreserved set becomes `%XX` with uppercase digits.
pub(super) fn encode_blob_name(name: &str) -> String {
    name.split('/')
        .map(encode_segment)
        .collect::<Vec<_>>()
        .join("/")
}

/// Percent-encodes one path segment.
pub(super) fn encode_segment(segment: &str) -> String {
    let mut encoded = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(char::from(byte));
            }
            _ => {
                encoded.push('%');
                encoded.push(char::from(HEX_DIGITS[usize::from(byte >> 4)]));
                encoded.push(char::from(HEX_DIGITS[usize::from(byte & 0x0f)]));
            }
        }
    }
    encoded
}

/// Rejects a container name that cannot appear in a path segment.
pub(super) fn validate_container(container: &str) -> Result<()> {
    if container.trim().is_empty() {
        return Err(validation("bucket", "must not be blank"));
    }
    if container.chars().any(|character| {
        matches!(character, '/' | '?' | '#' | '%') || character.is_control() || character == ' '
    }) {
        return Err(validation(
            "bucket",
            format!("`{container}` contains a character a container name cannot carry"),
        ));
    }
    Ok(())
}

/// Rejects a blob key that cannot be addressed as a path.
pub(super) fn validate_key(key: &str) -> Result<()> {
    if key.is_empty() {
        return Err(validation("key", "must not be empty"));
    }
    if key.starts_with('/') {
        return Err(validation("key", "must not start with `/`"));
    }
    if key.chars().any(char::is_control) {
        return Err(validation(
            "key",
            format!("`{key}` contains a control character"),
        ));
    }
    Ok(())
}

/// Normalizes a SAS token into a query string: no leading `?`, no surrounding whitespace.
pub(super) fn normalize_sas_token(token: &str) -> Result<String> {
    let token = token.trim().trim_start_matches('?').trim();
    if token.is_empty() {
        return Err(validation("token", "must not be blank"));
    }
    Ok(token.to_string())
}

/// Resolves the endpoint and its path prefix from `config`.
///
/// An empty endpoint is derived: from the account name when it is set, otherwise the Azurite
/// emulator address. The path component of the endpoint is kept as a prefix for every request
/// (a proxy mount or the Azurite `/{account}` path), so it is part of the signed path too.
pub(super) fn endpoint_parts(config: &AzureBlobConfig) -> Result<(String, String)> {
    let configured = config.endpoint.trim().trim_end_matches('/');
    let endpoint = if configured.is_empty() {
        let account = config.account.trim();
        if account.is_empty() {
            AZURITE_ENDPOINT.to_string()
        } else {
            format!("https://{account}.{DEFAULT_ENDPOINT_SUFFIX}")
        }
    } else {
        configured.to_string()
    };
    let base = validate_endpoint(&endpoint)?;
    let rest = base
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or_default();
    let path_prefix = rest
        .find('/')
        .map(|index| rest[index..].trim_end_matches('/').to_string())
        .unwrap_or_default();
    Ok((base, path_prefix))
}

/// Validates the endpoint and returns it without a trailing slash.
///
/// The endpoint must be absolute HTTP(S) with a host; a path is kept, because a proxy or the
/// Azurite account mount may live below one. Hand-rolled rather than parsed, since this crate
/// deliberately carries no URL crate.
pub(super) fn validate_endpoint(endpoint: &str) -> Result<String> {
    let trimmed = endpoint.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(Error::config("Azure Blob endpoint must not be empty"));
    }
    let Some((scheme, rest)) = trimmed.split_once("://") else {
        return Err(Error::config(format!(
            "invalid Azure Blob endpoint {endpoint:?}: expected an absolute http(s) URL"
        )));
    };
    if !matches!(scheme, "http" | "https") {
        return Err(Error::config(format!(
            "unsupported Azure Blob endpoint scheme {scheme:?}; expected `http` or `https`"
        )));
    }
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    if authority.is_empty() {
        return Err(Error::config(format!(
            "Azure Blob endpoint {endpoint:?} has no host"
        )));
    }
    Ok(trimmed.to_string())
}
