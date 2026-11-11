//! `SecretString`: a string wrapper that redacts itself in `Debug`/`Display`.
#![allow(clippy::module_name_repetitions)]

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::constants::REDACTED;
use crate::security::redact::constant_time_eq;

/// A string holding a credential that never leaks through `Debug`/`Display`.
///
/// Serialization keeps the plaintext value so configuration round-trips (Python
/// parity: secrets are part of the serialized config), while `Debug`, `Display`,
/// and every log line show [`REDACTED`] instead. Equality is a constant-time byte
/// comparison, so comparing a guess against the secret does not leak its prefix.
///
/// Use [`SecretString::expose_secret`] at the single point where the value is
/// actually needed (for example, building an `Authorization` header).
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(transparent)]
#[must_use]
pub struct SecretString(String);

impl SecretString {
    /// Wraps a value in a [`SecretString`].
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the plaintext secret.
    ///
    /// Call this only where the secret is consumed, never when formatting a log
    /// message or an error.
    #[must_use]
    pub fn expose_secret(&self) -> &str {
        &self.0
    }

    /// Returns `true` when the secret is empty (an unset credential).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the length of the secret in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SecretString(\"{REDACTED}\")")
    }
}

impl fmt::Display for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

impl PartialEq for SecretString {
    fn eq(&self, other: &Self) -> bool {
        constant_time_eq(self.0.as_bytes(), other.0.as_bytes())
    }
}

impl Eq for SecretString {}

impl From<String> for SecretString {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for SecretString {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn debug_and_display_redact_the_value() {
        let secret = SecretString::from("sk-super-secret");

        assert_eq!(format!("{secret:?}"), "SecretString(\"<REDACTED>\")");
        assert_eq!(secret.to_string(), "<REDACTED>");
        assert!(!format!("{secret:?}").contains("sk-super-secret"));
    }

    #[test]
    fn json_round_trip_keeps_the_plaintext() {
        let secret = SecretString::from("token-123");
        let json = serde_json::to_string(&secret).unwrap();
        assert_eq!(json, "\"token-123\"");

        let restored: SecretString = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.expose_secret(), "token-123");
        assert_eq!(restored, secret);
    }

    #[test]
    fn expose_secret_and_accessors() {
        let secret = SecretString::new("abc");
        assert_eq!(secret.expose_secret(), "abc");
        assert_eq!(secret.len(), 3);
        assert!(!secret.is_empty());

        let empty = SecretString::default();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.expose_secret(), "");
    }

    #[test]
    fn equality_is_value_based() {
        assert_eq!(SecretString::from("same"), SecretString::from("same"));
        assert_ne!(SecretString::from("same"), SecretString::from("other"));
        assert_ne!(SecretString::from("short"), SecretString::from("longer"));
    }
}
