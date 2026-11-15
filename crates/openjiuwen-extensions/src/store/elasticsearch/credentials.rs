//! Credential resolution shared by the Elasticsearch vector and document backends.
//!
//! Python parity: `openjiuwen/extensions/store/vector/es_vector_store.py` reads
//! HTTP basic credentials from the endpoint URL, while the kernel's
//! `ElasticsearchDocumentStore` additionally accepts explicit `username`,
//! `password`, and `api_key` configuration. Both Rust backends apply one rule
//! set, so this module owns it once:
//!
//! - The URL userinfo (`http://user:pass@host:9200`) supplies the basic-auth
//!   pair when the explicit fields do not.
//! - The explicit fields win field by field: an explicit user name replaces the
//!   userinfo user name, and an explicit password replaces the userinfo
//!   password.
//! - A user name without a password (from either source) sends an empty
//!   password.
//! - A password without a user name is kept but never sent, because HTTP basic
//!   authentication needs both halves.
//! - An API key is sent as `Authorization: ApiKey <key>`, which ranks below
//!   basic authentication, matching the kernel document store.

use url::Url;

/// The credentials an Elasticsearch backend resolved from its configuration.
///
/// [`ElasticsearchCredentials::resolve`] applies every precedence rule listed in
/// the module docs, so a caller only has to know whether a basic-auth pair or an
/// API key applies (see [`ElasticsearchCredentials::basic_auth`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ElasticsearchCredentials {
    /// User name of HTTP basic authentication, from the explicit field or the
    /// URL userinfo.
    pub(crate) username: Option<String>,
    /// Password of HTTP basic authentication; empty when a user name has no
    /// password.
    pub(crate) password: Option<String>,
    /// API key sent as `Authorization: ApiKey <key>` when no basic-auth pair
    /// applies.
    pub(crate) api_key: Option<String>,
}

impl ElasticsearchCredentials {
    /// Resolves the credentials of `parsed` plus the explicit configuration
    /// fields.
    ///
    /// See the module docs for the exact precedence rules; no field is
    /// validated here, so a blank explicit value is honoured as given (the
    /// configuration decoders turn a blank string into `None` before this runs).
    pub(crate) fn resolve(
        parsed: &Url,
        username: Option<&str>,
        password: Option<&str>,
        api_key: Option<&str>,
    ) -> Self {
        let userinfo_username =
            (!parsed.username().is_empty()).then(|| parsed.username().to_string());
        let userinfo_password = parsed.password().map(str::to_string);
        let username = username.map(str::to_string).or(userinfo_username);
        let password = match (password.map(str::to_string), username.is_some()) {
            (Some(explicit), _) => Some(explicit),
            (None, true) => Some(userinfo_password.unwrap_or_default()),
            (None, false) => None,
        };
        Self {
            username,
            password,
            api_key: api_key.map(str::to_string),
        }
    }

    /// The HTTP basic authentication pair, or `None` when either half is
    /// missing.
    pub(crate) fn basic_auth(&self) -> Option<(&str, &str)> {
        match (&self.username, &self.password) {
            (Some(username), Some(password)) => Some((username, password)),
            _ => None,
        }
    }
}

/// The endpoint without userinfo, query, fragment, or trailing slash.
pub(crate) fn endpoint_without_userinfo(parsed: &Url) -> String {
    let mut stripped = parsed.clone();
    let _ = stripped.set_username("");
    let _ = stripped.set_password(None);
    stripped.set_query(None);
    stripped.set_fragment(None);
    stripped.as_str().trim_end_matches('/').to_string()
}
