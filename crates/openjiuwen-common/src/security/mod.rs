//! Secrets, redaction, and constant-time comparison helpers.
//!
//! Port of the Python package `openjiuwen.core.common.security` together with
//! `utils/url_utils.py` and `utils/header_utils.py`. Everything here exists so that
//! credentials never reach a log sink: [`SecretString`] redacts itself in
//! `Debug`/`Display`, [`redact_json`] masks known sensitive keys, and
//! [`redact_url_password`] masks credentials embedded in connection URLs.

pub mod redact;
pub mod secret;

pub use redact::{
    constant_time_eq, redact_json, redact_str, redact_url_password, sanitize_headers,
    DEFAULT_SENSITIVE_FIELDS, REDACTED,
};
pub use secret::SecretString;
