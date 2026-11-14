//! [`AuthScheme`] and [`resolve_auth`]: how a vendor attaches its credential.
//!
//! The scheme is data, never the secret: the provider reads
//! [`ProviderConfig::api_key`](super::super::registry::ProviderConfig::api_key) at request
//! time, so a resolved scheme can be logged and printed.
use serde_json::Value;

use super::super::registry::ProviderConfig;

/// How a vendor authenticates a request.
///
/// The variant is the vendor default or the caller's `options["auth"]` override; see
/// [`resolve_auth`]. The credential itself is never part of this value — it is read from
/// [`ProviderConfig::api_key`] at request time, so a scheme can be logged.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AuthScheme {
    /// No credential is attached.
    #[default]
    None,
    /// `Authorization: Bearer <api_key>`.
    Bearer,
    /// `Authorization: Token <api_key>`.
    Token,
    /// The raw `api_key` in the named header.
    Header {
        /// Header name, for example `X-Api-Key`.
        name: String,
    },
    /// The raw `api_key` in the named query parameter.
    Query {
        /// Query parameter name, for example `api_key`.
        name: String,
    },
}

impl AuthScheme {
    /// The header this scheme injects for `api_key`, if any.
    #[must_use]
    pub fn header_for(&self, api_key: &str) -> Option<(String, String)> {
        if api_key.is_empty() {
            return None;
        }
        match self {
            Self::Bearer => Some(("Authorization".to_owned(), format!("Bearer {api_key}"))),
            Self::Token => Some(("Authorization".to_owned(), format!("Token {api_key}"))),
            Self::Header { name } => Some((name.clone(), api_key.to_owned())),
            Self::None | Self::Query { .. } => None,
        }
    }

    /// The query parameter this scheme injects for `api_key`, if any.
    #[must_use]
    pub fn query_for(&self, api_key: &str) -> Option<(String, String)> {
        if api_key.is_empty() {
            return None;
        }
        match self {
            Self::Query { name } => Some((name.clone(), api_key.to_owned())),
            Self::None | Self::Bearer | Self::Token | Self::Header { .. } => None,
        }
    }
}

/// Resolves the effective authentication scheme.
///
/// `default` is the vendor default; `config.option("auth")` overrides it
/// (`none`, `bearer`, `token`, `header`, `query`, ASCII-case-insensitive and trimmed).
/// `header` takes its name from `options["auth_header"]` (default `Authorization`) and
/// `query` from `options["auth_query_param"]` (default `api_key`). An unknown, blank, or
/// non-string `auth` value keeps `default`: a typo must never silently disable
/// authentication *and* never fail a config load.
#[must_use]
pub fn resolve_auth(default: AuthScheme, config: &ProviderConfig) -> AuthScheme {
    let Some(raw) = config.option("auth").and_then(Value::as_str) else {
        return default;
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" => AuthScheme::None,
        "bearer" => AuthScheme::Bearer,
        "token" => AuthScheme::Token,
        "header" => AuthScheme::Header {
            name: option_text(config, "auth_header").unwrap_or_else(|| "Authorization".to_owned()),
        },
        "query" => AuthScheme::Query {
            name: option_text(config, "auth_query_param").unwrap_or_else(|| "api_key".to_owned()),
        },
        _ => default,
    }
}

/// Reads one trimmed, non-blank string option.
fn option_text(config: &ProviderConfig, key: &str) -> Option<String> {
    config
        .option(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}
