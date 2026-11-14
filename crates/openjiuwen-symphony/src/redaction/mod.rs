//! Credential and PII redaction for model-facing and packaged text.
//!
//! Ports the two Python privacy modules that guard Symphony's public JSON
//! contracts and its distilled artifacts:
//! `openjiuwen.symphony.models._redaction` (`normalize_sensitive_name`,
//! `is_sensitive_name`, `redact_sensitive_text`, `redact_sensitive_json`) and
//! `openjiuwen.symphony.flow.privacy` (`redact_common_credentials`,
//! `sanitize_distilled_text`, `redact_private_json`).
//!
//! # Intent
//!
//! Redaction is conservative and offline: a recognized secret is replaced by a
//! fixed marker, everything around it is preserved byte for byte, and nothing
//! depends on the network, the clock, or a model. The scanner is hand rolled
//! (Python's patterns use lookbehind assertions, which the `regex` crate cannot
//! express) and only ever slices text at character boundaries, so non-ASCII
//! input can never panic.
//!
//! # Pattern set
//!
//! [`redact_sensitive_text`] runs these passes in order, replacing only the
//! matched span:
//!
//! 1. `-----BEGIN … PRIVATE KEY-----` … `-----END … PRIVATE KEY-----` blocks;
//! 2. `Bearer <token>` (case-insensitive, at least 8 token characters);
//! 3. JWTs (`eyJ`-headed header and payload segments plus a signature segment);
//! 4. OpenAI-style `sk-…` keys;
//! 5. AWS access key ids (`AKIA`/`ASIA` plus 16 uppercase alphanumerics);
//! 6. `key=value` and `"key": "value"` assignments whose key
//!    [`is_sensitive_name`] recognizes.
//!
//! Every pass is idempotent: redacting an already redacted text returns it
//! unchanged. [`sanitize_distilled_text`] adds the two flow-level passes Python
//! applies to model-derived prose (verbatim request replacement and a length
//! cut), and [`redact_sensitive_json`]/[`redact_private_json`] are the JSON
//! entry points used when a metadata blob is embedded in a recipe or package.
//!
//! # Example
//!
//! ```
//! use openjiuwen_symphony::redaction::{redact_sensitive_text, sanitize_distilled_text};
//!
//! let text = redact_sensitive_text("api_key: sk-live-0123456789abcdef");
//! assert_eq!(text, "api_key=<redacted>");
//! assert_eq!(
//!     sanitize_distilled_text("password = hunter2", None),
//!     "password=<redacted>"
//! );
//! ```

mod json;
mod names;
mod scanner;

pub use json::{redact_private_json, redact_sensitive_json};
pub use names::{is_sensitive_name, normalize_sensitive_name};
pub(crate) use scanner::sanitize_with_queries;
pub use scanner::{redact_common_credentials, redact_sensitive_text, sanitize_distilled_text};

/// The marker that replaces a redacted value.
pub const REDACTED: &str = "<redacted>";
/// The marker that replaces a bare credential (Python's `<redacted-credential>`).
pub const REDACTED_CREDENTIAL: &str = "<redacted-credential>";
/// The marker that replaces a verbatim source request (Python's `<redacted-request>`).
pub const REDACTED_REQUEST: &str = "<redacted-request>";
/// The character limit [`sanitize_distilled_text`] truncates to.
///
/// Mirrors Python's `_MAX_DISTILLED_TEXT_LENGTH`.
pub const DISTILLED_TEXT_LIMIT: usize = 2_000;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn markers_are_stable() {
        assert_eq!(REDACTED, "<redacted>");
        assert_eq!(REDACTED_CREDENTIAL, "<redacted-credential>");
        assert_eq!(REDACTED_REQUEST, "<redacted-request>");
        assert_eq!(DISTILLED_TEXT_LIMIT, 2_000);
    }

    #[test]
    fn the_documented_example_holds() {
        assert_eq!(
            redact_sensitive_text("api_key: sk-live-0123456789abcdef"),
            "api_key=<redacted>"
        );
        assert_eq!(
            sanitize_distilled_text("password = hunter2", None),
            "password=<redacted>"
        );
    }
}
