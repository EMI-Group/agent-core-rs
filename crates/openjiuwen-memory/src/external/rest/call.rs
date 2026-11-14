//! [`RestCall`]: one vendor request, relative to the base URL in use.
//!
//! A spec never sees an endpoint, so the same call can be replayed against a failover
//! base; [`RestCall::absolute_url`] is the only place that joins the two.
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::Method;
use serde::Serialize;
use serde_json::Value;

/// One vendor request, relative to whichever base URL the provider is currently using.
///
/// The path is intentionally base-less: the same call is replayed against a failover
/// endpoint unchanged. Query entries and headers are ordered lists so a spec keeps full
/// control; the provider appends authentication and, for a JSON body, serializes it on
/// every attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct RestCall {
    /// HTTP verb, e.g. `Method::POST`.
    pub method: Method,
    /// Path relative to the endpoint (`"/items"` and `"items"` are equivalent).
    pub path: String,
    /// Query parameters appended in order.
    pub query: Vec<(String, String)>,
    /// Extra headers for this call only; the provider's vendor headers are added too.
    pub headers: Vec<(String, String)>,
    /// Structured body; serialized into a JSON payload when the request is sent.
    pub body: Option<Value>,
}

impl RestCall {
    /// Builds a call with `method` and `path` and nothing else.
    #[must_use]
    pub fn new(method: Method, path: impl Into<String>) -> Self {
        Self {
            method,
            path: path.into(),
            query: Vec::new(),
            headers: Vec::new(),
            body: None,
        }
    }

    /// Builds a `GET` call.
    #[must_use]
    pub fn get(path: impl Into<String>) -> Self {
        Self::new(Method::GET, path)
    }

    /// Builds a `POST` call.
    #[must_use]
    pub fn post(path: impl Into<String>) -> Self {
        Self::new(Method::POST, path)
    }

    /// Builds a `PUT` call.
    #[must_use]
    pub fn put(path: impl Into<String>) -> Self {
        Self::new(Method::PUT, path)
    }

    /// Builds a `PATCH` call.
    #[must_use]
    pub fn patch(path: impl Into<String>) -> Self {
        Self::new(Method::PATCH, path)
    }

    /// Builds a `DELETE` call.
    #[must_use]
    pub fn delete(path: impl Into<String>) -> Self {
        Self::new(Method::DELETE, path)
    }

    /// Appends one query parameter.
    #[must_use]
    pub fn query(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query.push((key.into(), value.into()));
        self
    }

    /// Appends several query parameters.
    #[must_use]
    pub fn queries(mut self, params: impl IntoIterator<Item = (String, String)>) -> Self {
        self.query.extend(params);
        self
    }

    /// Appends one header for this call.
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Sets the JSON body of this call.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the value cannot be serialized.
    pub fn json(mut self, body: &impl Serialize) -> Result<Self> {
        self.body = Some(serde_json::to_value(body).map_err(|error| {
            Error::validation(format!(
                "serializing a memory provider request body: {error}"
            ))
        })?);
        Ok(self)
    }

    /// Sets a body that is already a JSON value.
    #[must_use]
    pub fn with_body(mut self, body: Value) -> Self {
        self.body = Some(body);
        self
    }

    /// Joins `base` and `path` with exactly one `/`.
    ///
    /// A base may end with a `/` and a path may start with one; both are normalised. An
    /// already-absolute path is returned unchanged, and a blank base leaves the path as
    /// it is.
    #[must_use]
    pub fn absolute_url(base: &str, path: &str) -> String {
        let base = base.trim();
        let path = path.trim();
        if path.starts_with("http://") || path.starts_with("https://") {
            return path.to_owned();
        }
        if base.is_empty() {
            return path.to_owned();
        }
        if path.is_empty() {
            return base.trim_end_matches('/').to_owned();
        }
        format!(
            "{}/{}",
            base.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }
}
