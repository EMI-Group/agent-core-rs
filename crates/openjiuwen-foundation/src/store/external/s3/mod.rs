//! S3-compatible object store with an in-crate AWS Signature Version 4 signer.
//!
//! [`S3ObjectStore`] implements the [`ObjectStore`](crate::store::object::ObjectStore)
//! contract of [`crate::store::object`] on top of any S3-compatible service —
//! AWS S3, `MinIO`, `Ceph`, `LocalStack`, an S3 gateway. Every request is signed by
//! [`sigv4`], which derives AWS Signature Version 4 from `sha2` directly, so no
//! vendor SDK (and no `hmac`/`hex` crate) is involved.
//!
//! # Configuration
//!
//! [`S3Config`] follows the convention of the other external backends: it is
//! `serde`-deserializable with a default for every field, so a store can be built
//! straight from a JSON object, and exposes builder setters for code.
//! Credentials come from [`Credentials`] and are required: signing cannot express
//! an anonymous request, so a store with an empty access key or secret fails at
//! request time with [`Error::Config`](openjiuwen_common::error::Error) instead
//! of sending an unsigned call.
//!
//! # Addressing
//!
//! `bucket` is always a method argument of the trait. [`S3Config::bucket`] is
//! kept for configuration parity and as a documented *default* bucket; the
//! [`ObjectStore`](crate::store::object::ObjectStore) contract always names a
//! bucket, so every call uses its own argument.
//!
//! With `path_style` (the default, and what `MinIO` and most gateways expect) a key
//! lives at `{endpoint}/{bucket}/{key}`. With `path_style = false` the bucket
//! becomes a sub-domain: `{scheme}://{bucket}.{host}[:{port}]/{key}`. A bucket
//! name containing dots is used as-is, which means `https://` against such a
//! bucket only validates with a wildcard certificate — a service-side property
//! this client cannot work around.
//!
//! # Differences from the reference backend
//!
//! - S3 reports a missing bucket and a missing key identically for `GET`, `HEAD`
//!   and `DELETE` (HTTP 404), so `get` returns `Ok(None)`, `exists` returns
//!   `Ok(false)` and `delete` is a no-op in both cases. Only `list` can tell the
//!   two apart through the error body, and it maps a 404 to
//!   [`Error::NotFound`](openjiuwen_common::error::Error).
//! - `list` follows `ListObjectsV2` continuation tokens, so a listing larger than
//!   one page is still returned whole (or up to `limit`).
//! - `list_buckets` lists the buckets the credentials can see, not the buckets of
//!   this store.

pub mod sigv4;

mod store;

#[cfg(test)]
mod tests;

use std::time::Duration;

use serde::{Deserialize, Serialize};
use url::Url;

use openjiuwen_common::error::{Error, Result};

pub use sigv4::Credentials;
pub use store::S3ObjectStore;

/// Region used when the configuration or the environment does not set one.
pub const DEFAULT_REGION: &str = "us-east-1";

/// Timeout used when the configuration does not set one.
const fn default_timeout() -> Duration {
    Duration::from_secs(30)
}

/// Configuration of an [`S3ObjectStore`].
///
/// ```
/// use openjiuwen_foundation::store::external::s3::{Credentials, S3Config};
///
/// let config = S3Config::new("http://localhost:9000")
///     .with_region("eu-central-1")
///     .with_credentials(Credentials::new("minio", "minio123"))
///     .with_bucket("uploads");
///
/// assert_eq!(config.region, "eu-central-1");
/// assert_eq!(config.bucket.as_deref(), Some("uploads"));
/// assert!(config.path_style);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct S3Config {
    /// Base URL of the service, for example `http://localhost:9000`.
    pub endpoint: String,
    /// AWS region of the signature scope.
    pub region: String,
    /// Default bucket; see the [module docs](self) for why calls always pass one.
    pub bucket: Option<String>,
    /// Credentials every request is signed with.
    pub credentials: Credentials,
    /// Address keys as `{endpoint}/{bucket}/{key}` instead of a bucket sub-domain.
    pub path_style: bool,
    /// Timeout of a single HTTP request.
    pub timeout: Duration,
}

impl Default for S3Config {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            region: DEFAULT_REGION.to_string(),
            bucket: None,
            credentials: Credentials::default(),
            path_style: true,
            timeout: default_timeout(),
        }
    }
}

impl S3Config {
    /// Creates a configuration pointing at `endpoint` and using every default.
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            ..Self::default()
        }
    }

    /// Sets the region of the signature scope.
    #[must_use]
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }

    /// Sets the credentials every request is signed with.
    #[must_use]
    pub fn with_credentials(mut self, credentials: Credentials) -> Self {
        self.credentials = credentials;
        self
    }

    /// Chooses between path-style and virtual-host addressing.
    #[must_use]
    pub const fn with_path_style(mut self, path_style: bool) -> Self {
        self.path_style = path_style;
        self
    }

    /// Sets the timeout of a single request.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the default bucket name.
    #[must_use]
    pub fn with_bucket(mut self, bucket: impl Into<String>) -> Self {
        self.bucket = Some(bucket.into());
        self
    }

    /// Builds a configuration from the environment.
    ///
    /// | Field | Variables, first non-empty wins |
    /// | --- | --- |
    /// | endpoint | `OPENJIUWEN_S3_ENDPOINT`, `AWS_ENDPOINT_URL`, `AWS_ENDPOINT_URL_S3` |
    /// | region | `OPENJIUWEN_S3_REGION`, `AWS_REGION`, `AWS_DEFAULT_REGION` |
    /// | access key id | `OPENJIUWEN_S3_ACCESS_KEY_ID`, `AWS_ACCESS_KEY_ID` |
    /// | secret access key | `OPENJIUWEN_S3_SECRET_ACCESS_KEY`, `AWS_SECRET_ACCESS_KEY` |
    /// | session token | `AWS_SESSION_TOKEN` |
    /// | bucket | `OPENJIUWEN_S3_BUCKET` |
    ///
    /// Empty variables count as unset. Validation happens in
    /// [`S3ObjectStore::new`], so this constructor cannot fail.
    #[must_use]
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Some(endpoint) = first_env(&[
            "OPENJIUWEN_S3_ENDPOINT",
            "AWS_ENDPOINT_URL",
            "AWS_ENDPOINT_URL_S3",
        ]) {
            config.endpoint = endpoint;
        }
        if let Some(region) =
            first_env(&["OPENJIUWEN_S3_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"])
        {
            config.region = region;
        }
        let access_key_id = first_env(&["OPENJIUWEN_S3_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"]);
        let secret_access_key =
            first_env(&["OPENJIUWEN_S3_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"]);
        if access_key_id.is_some() || secret_access_key.is_some() {
            config.credentials.access_key_id = access_key_id.unwrap_or_default();
            if let Some(secret) = secret_access_key {
                config.credentials.secret_access_key = secret.into();
            }
        }
        if let Some(token) = first_env(&["AWS_SESSION_TOKEN"]) {
            config.credentials.session_token = Some(token.into());
        }
        if let Some(bucket) = first_env(&["OPENJIUWEN_S3_BUCKET"]) {
            config.bucket = Some(bucket);
        }
        config
    }
}

/// Parses `endpoint` into a base URL, rejecting anything that is not absolute HTTP(S).
///
/// The path of the endpoint (if any) is preserved as a prefix for every request.
pub(crate) fn parse_endpoint(endpoint: &str) -> Result<Url> {
    let endpoint = endpoint.trim();
    if endpoint.is_empty() {
        return Err(Error::config("S3 endpoint must not be empty"));
    }
    let url = Url::parse(endpoint)
        .map_err(|error| Error::config(format!("invalid S3 endpoint {endpoint:?}: {error}")))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(Error::config(format!(
            "unsupported S3 endpoint scheme {:?}; expected `http` or `https`",
            url.scheme()
        )));
    }
    if url.host_str().is_none_or(str::is_empty) {
        return Err(Error::config(format!(
            "S3 endpoint {endpoint:?} has no host"
        )));
    }
    Ok(url)
}

/// The first environment variable in `names` that is set to a non-empty value.
fn first_env(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    })
}
