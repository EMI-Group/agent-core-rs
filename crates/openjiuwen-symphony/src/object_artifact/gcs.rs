//! Google Cloud Storage object backend over the GCS JSON API.
//!
//! [`GcsObjectStore`] implements the
//! [`ObjectStore`](openjiuwen_foundation::store::ObjectStore) contract on top of the GCS JSON
//! API, so a capability graph can live in a Google bucket exactly like it lives in an
//! S3-compatible one:
//! [`ObjectGraphArtifactStore::gcs`](crate::object_artifact::ObjectGraphArtifactStore::gcs) or
//! [`ArtifactTransport::Gcs`](crate::object_artifact::ArtifactTransport) both build this
//! transport.
//!
//! # Authentication
//!
//! Three modes, tried in this order:
//!
//! 1. a non-empty [`GcsConfig::access_token`] is sent as `Authorization: Bearer <token>`, and it
//!    always wins — an explicitly configured token is never overridden;
//! 2. otherwise a service-account key attached by
//!    [`from_service_account_file_with`](crate::object_artifact::GcsObjectStore::from_service_account_file_with)
//!    mints an OAuth 2.0 access token: an RS256 JWT assertion is posted to the key file's
//!    `token_uri`, and the returned token is sent as the bearer token. Minting is lazy (the
//!    first request that needs a token triggers it) and the token is cached until 60 seconds
//!    before it expires, so a burst of object operations signs one assertion;
//! 3. otherwise requests are anonymous, which is what a local emulator (`fake-gcs-server`, the
//!    GCS emulator in `gcloud`) and a public read-only bucket need.
//!
//! The key material never leaves the process: neither the PEM, the assertion nor a minted token
//! appears in a `Debug` render or an error message. A failed mint is a typed
//! [`Error::External`] naming the token endpoint — never a silent fallback to anonymous
//! requests, which would surface as a confusing permission error.
//!
//! The scope a minted token carries is `https://www.googleapis.com/auth/devstorage.read_write`
//! (`devstorage.read_write`, since the transport publishes revisions as well as reads them).
//!
//! # Endpoints
//!
//! `endpoint` is the API host ([`DEFAULT_ENDPOINT`] by default) and may carry a path, which is
//! preserved as a prefix for every request (a proxy or an emulator mounted below a
//! sub-path). The object endpoints are:
//!
//! | Operation | Request |
//! | --- | --- |
//! | `create_bucket` | `POST {endpoint}/storage/v1/b?project=…` with `{"name": "<bucket>"}` |
//! | `delete_bucket` | `DELETE {endpoint}/storage/v1/b/{bucket}` |
//! | `list_buckets` | `GET {endpoint}/storage/v1/b?project=…` |
//! | `put` | `POST {endpoint}/upload/storage/v1/b/{bucket}/o?uploadType=media&name={key}` |
//! | `get` | `GET {endpoint}/storage/v1/b/{bucket}/o/{key}?alt=media` |
//! | `exists` | `GET {endpoint}/storage/v1/b/{bucket}/o/{key}` (the object resource) |
//! | `delete` | `DELETE {endpoint}/storage/v1/b/{bucket}/o/{key}` |
//! | `list` | `GET {endpoint}/storage/v1/b/{bucket}/o?prefix={prefix}&maxResults={n}` |
//!
//! An object name is a single path segment, so `/` inside a name is percent-encoded (`%2F`)
//! rather than kept as a separator; [`encode_object_name`] is the encoder, and it leaves only
//! the RFC 3986 unreserved characters alone.
//!
//! # Missing buckets and missing objects
//!
//! GCS answers both with HTTP 404, and `get` has to tell them apart to honour the
//! [`ObjectStore`](openjiuwen_foundation::store::ObjectStore) contract (a missing object is
//! `Ok(None)`, a missing bucket is [`Error::NotFound`]). The reply's error message is the only
//! signal: the bucket case says the bucket does not exist. That check is best effort, so an
//! emulator that answers a bare 404 is read as a missing object. `list` maps every 404 to a
//! missing bucket, because a listing names no object, and `exists`/`delete` treat every 404 as
//! "absent" (`false` and a no-op) — the same tolerance the S3 backend has, where the two cases
//! are indistinguishable.
//!
//! # Timestamps
//!
//! GCS reports `size` as a decimal string and `updated` as RFC 3339. The resource is
//! normalized into the [`ObjectMeta`] document shape and decoded by [`ObjectMeta`]'s own
//! `serde` implementation, which parses the timestamp — so this module needs no date-time
//! library of its own. A missing or malformed field falls back to the uploaded payload length
//! and the current instant.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, HttpResponse};
use openjiuwen_common::security::SecretString;
use openjiuwen_common::utils::id::now_utc;
use openjiuwen_foundation::store::{ObjectMeta, ObjectStore};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::http::failure;
use crate::error::{validation, Error, Result};

mod key;
mod signing;
mod token;

/// Endpoint used when the configuration does not set one.
pub const DEFAULT_ENDPOINT: &str = "https://storage.googleapis.com";

/// The credential-file forms a GCS key file is still refused for.
///
/// [`GcsObjectStore::from_service_account_file`] reads a `service_account` JSON key file and
/// exchanges it for an OAuth 2.0 access token through the file's `token_uri`. Other credential
/// files — workload identity federation (`type: external_account`), `gcloud` user credentials
/// (`type: authorized_user`), the GCE metadata server and `GOOGLE_APPLICATION_CREDENTIALS`
/// auto-discovery — are not read, so they are refused rather than silently sent anonymously.
/// A service-account key file, an OAuth 2.0 access token, or an already provisioned bucket
/// covers every GCS artifact path.
pub const SERVICE_ACCOUNT_REASON: &str = "Google Cloud Storage reads `service_account` JSON key \
     files only; workload identity federation (`external_account`), `authorized_user` \
     credentials, the GCE metadata server and GOOGLE_APPLICATION_CREDENTIALS auto-discovery are \
     not read — supply a service-account key file or an OAuth 2.0 access token";

/// Short service name used in diagnostics.
const SERVICE: &str = "GCS";

/// Path segment of the bucket and object resource endpoints.
const JSON_PATH: &str = "storage/v1";

/// Path segment of the upload endpoint.
const UPLOAD_PATH: &str = "upload/storage/v1";

/// Maximum number of objects GCS returns in one page.
const MAX_RESULTS_PER_PAGE: usize = 1000;

/// Hard cap on the pages one `list` call follows, so a misbehaving service cannot make the
/// call loop forever.
const MAX_PAGES: usize = 1000;

/// Uppercase hexadecimal digits of a percent escape.
const HEX_DIGITS: &[u8; 16] = b"0123456789ABCDEF";

/// Timeout used when the configuration does not set one.
const fn default_timeout() -> Duration {
    Duration::from_secs(60)
}

/// Configuration of a [`GcsObjectStore`].
///
/// Every field has a default, so a configuration round-trips through JSON and a partial
/// document keeps the intended values. [`GcsConfig::from_env`] reads the conventional
/// variables, and the `with_*` setters cover code.
///
/// ```
/// use openjiuwen_symphony::object_artifact::GcsConfig;
///
/// let config = GcsConfig::default()
///     .with_project("demo-project")
///     .with_access_token("ya29.token");
///
/// assert_eq!(config.endpoint, "https://storage.googleapis.com");
/// assert_eq!(config.project.as_deref(), Some("demo-project"));
/// assert!(!format!("{config:?}").contains("ya29.token"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct GcsConfig {
    /// Base URL of the JSON API; a path is kept as a prefix for every request.
    pub endpoint: String,
    /// Google Cloud project, required by `create_bucket` and `list_buckets` only.
    ///
    /// GCS cannot create or enumerate buckets without a project, so those two operations
    /// report [`Error::Validation`] naming this field when it is unset.
    pub project: Option<String>,
    /// OAuth 2.0 access token sent as `Authorization: Bearer <token>`.
    ///
    /// `None` (or an empty token) sends anonymous requests. The token serializes so the
    /// configuration round-trips, but `Debug` redacts it.
    pub access_token: Option<SecretString>,
    /// Timeout of a single HTTP request.
    pub timeout: Duration,
}

impl Default for GcsConfig {
    fn default() -> Self {
        Self {
            endpoint: DEFAULT_ENDPOINT.to_string(),
            project: None,
            access_token: None,
            timeout: default_timeout(),
        }
    }
}

impl GcsConfig {
    /// Creates a configuration pointing at `endpoint` and using every other default.
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            ..Self::default()
        }
    }

    /// Sets the base URL of the JSON API.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Sets the Google Cloud project used to create and list buckets.
    #[must_use]
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Sets the OAuth 2.0 access token every request is authenticated with.
    #[must_use]
    pub fn with_access_token(mut self, token: impl Into<SecretString>) -> Self {
        self.access_token = Some(token.into());
        self
    }

    /// Sets the timeout of a single request.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Builds a configuration from the environment.
    ///
    /// | Field | Variables, first non-empty wins |
    /// | --- | --- |
    /// | endpoint | `OPENJIUWEN_GCS_ENDPOINT`, `GCS_ENDPOINT` |
    /// | project | `OPENJIUWEN_GCS_PROJECT`, `GOOGLE_CLOUD_PROJECT` |
    /// | access token | `OPENJIUWEN_GCS_ACCESS_TOKEN`, `GOOGLE_OAUTH_ACCESS_TOKEN` |
    ///
    /// Empty variables count as unset. Validation happens in [`GcsObjectStore::new`], so this
    /// constructor cannot fail.
    #[must_use]
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Some(endpoint) = first_env(&["OPENJIUWEN_GCS_ENDPOINT", "GCS_ENDPOINT"]) {
            config.endpoint = endpoint;
        }
        if let Some(project) = first_env(&["OPENJIUWEN_GCS_PROJECT", "GOOGLE_CLOUD_PROJECT"]) {
            config.project = Some(project);
        }
        if let Some(token) =
            first_env(&["OPENJIUWEN_GCS_ACCESS_TOKEN", "GOOGLE_OAUTH_ACCESS_TOKEN"])
        {
            config.access_token = Some(token.into());
        }
        config
    }
}

/// An [`ObjectStore`](openjiuwen_foundation::store::ObjectStore) backed by Google Cloud
/// Storage.
///
/// The store holds one HTTP client and one validated base URL, so it is cheap to share behind
/// an `Arc` and every method takes `&self`.
///
/// # Errors
///
/// [`GcsObjectStore::new`] validates the endpoint. Every operation returns the errors
/// documented on [`ObjectStore`](openjiuwen_foundation::store::ObjectStore), plus
/// [`Error::Validation`] for a blank or malformed bucket name or object key,
/// [`Error::Config`] for a project-less bucket operation, and an [`Error::External`] naming the
/// operation, the status, the URL and a body excerpt when the service rejects a request.
#[derive(Debug, Clone)]
pub struct GcsObjectStore {
    /// HTTP client shared by every request.
    client: HttpClient,
    /// Configuration the store was built from.
    config: GcsConfig,
    /// Validated API root: the endpoint without a trailing slash.
    base: String,
    /// Service-account credentials, when the store mints its own access tokens.
    service_account: Option<Arc<token::TokenSource>>,
}

impl GcsObjectStore {
    /// Builds a store from `config`.
    ///
    /// Redirects stay enabled (the default), because the JSON API may redirect a media
    /// download to a signed URL.
    ///
    /// The store authenticates with [`GcsConfig::access_token`] when it holds one and stays
    /// anonymous otherwise; use [`GcsObjectStore::from_service_account_file_with`] to mint
    /// tokens from a service-account key file instead.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when `config.endpoint` is not an absolute HTTP(S) URL, and
    /// propagates the HTTP client's configuration errors.
    pub fn new(config: GcsConfig) -> Result<Self> {
        let base = validate_endpoint(&config.endpoint)?;
        let client = HttpClient::new(HttpClientConfig::default().with_timeout(config.timeout))?;
        Ok(Self {
            client,
            config,
            base,
            service_account: None,
        })
    }

    /// Returns the configuration the store was built from.
    #[must_use]
    pub const fn config(&self) -> &GcsConfig {
        &self.config
    }

    /// Builds a store that mints OAuth 2.0 access tokens from a service-account key file.
    ///
    /// `path` is the JSON key file `gcloud iam service-accounts keys create` writes. The key is
    /// read once, here, and never again: every request that needs a token signs an RS256 JWT
    /// assertion with the in-memory key and posts it to the file's `token_uri`, and the token
    /// that comes back is cached until 60 seconds before it expires. Nothing leaves the
    /// process except the assertion — the PEM is never transmitted, logged or rendered.
    ///
    /// The store targets the public endpoint ([`DEFAULT_ENDPOINT`]) and takes its project from
    /// the key file; [`from_service_account_file_with`](Self::from_service_account_file_with)
    /// overrides both.
    ///
    /// ```no_run
    /// use openjiuwen_symphony::object_artifact::GcsObjectStore;
    ///
    /// let store = GcsObjectStore::from_service_account_file("/keys/service-account.json")?;
    /// # Ok::<(), openjiuwen_symphony::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::External`] naming `path` when the file cannot be read, [`Error::Config`] when it
    /// is not a service-account key file or names an unusable endpoint, and
    /// [`Error::Validation`] naming the offending field of a malformed key file.
    pub fn from_service_account_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_service_account_file_with(GcsConfig::default(), path)
    }

    /// Builds a store from `config` that mints access tokens from `path`.
    ///
    /// The key file is parsed before the configuration is validated, so a malformed key is
    /// reported as such rather than as an endpoint problem. `config.project` falls back to the
    /// key file's `project_id` when it is unset, which is what `create_bucket`/`list_buckets`
    /// need. Authentication follows the documented precedence: a non-empty
    /// [`GcsConfig::access_token`] still wins over the key file.
    ///
    /// # Errors
    ///
    /// The key-file errors of [`from_service_account_file`](Self::from_service_account_file)
    /// plus the configuration errors of [`GcsObjectStore::new`].
    pub fn from_service_account_file_with(
        config: GcsConfig,
        path: impl AsRef<Path>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let key = key::ServiceAccountKey::from_file(path)?;
        let mut config = config;
        if config.project.is_none() {
            config.project = key.project_id().map(str::to_string);
        }
        let mut store = Self::new(config)?;
        store.service_account = Some(Arc::new(token::TokenSource::new(key)));
        Ok(store)
    }

    /// The URL of `bucket`, or of `key` inside it.
    fn bucket_url(&self, bucket: &str) -> Result<String> {
        validate_bucket(bucket)?;
        Ok(format!("{}/{JSON_PATH}/b/{bucket}", self.base))
    }

    /// The URL of one object resource.
    fn object_url(&self, bucket: &str, key: &str) -> Result<String> {
        validate_key(key)?;
        Ok(format!(
            "{}/o/{}",
            self.bucket_url(bucket)?,
            encode_object_name(key)
        ))
    }

    /// Sends `request` with the configured timeout and the configured authentication.
    ///
    /// The precedence is deterministic and documented on the module: a non-empty
    /// [`GcsConfig::access_token`] wins, then a service-account key mints (or reuses) a bearer
    /// token, and only a store with neither sends an anonymous request.
    ///
    /// The response is not interpreted: each caller decides whether a non-2xx status is a
    /// tolerated absence or a failure.
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse> {
        let mut request = request.timeout(self.config.timeout);
        if let Some(token) = self.explicit_token() {
            request = request.bearer(token);
        } else if let Some(service_account) = &self.service_account {
            let token = service_account
                .access_token(&self.client, self.config.timeout)
                .await?;
            request = request.bearer(token);
        }
        self.client.execute(request).await
    }

    /// The explicitly configured access token, when it is set and non-empty.
    fn explicit_token(&self) -> Option<SecretString> {
        self.config
            .access_token
            .as_ref()
            .filter(|token| !token.is_empty())
            .cloned()
    }

    /// Sends `request` and maps any non-2xx status to an error naming `operation`.
    async fn send_ok(&self, request: HttpRequest, operation: &str) -> Result<HttpResponse> {
        let response = self.send(request).await?;
        if response.is_success() {
            Ok(response)
        } else {
            Err(failure(SERVICE, operation, &response))
        }
    }

    /// Returns the `project` of the configuration, or a validation error naming it.
    fn project(&self, operation: &str) -> Result<&str> {
        self.config
            .project
            .as_deref()
            .map(str::trim)
            .filter(|project| !project.is_empty())
            .ok_or_else(|| {
                validation(
                    "project",
                    format!(
                        "GCS cannot `{operation}` a bucket without a Google Cloud project; \
                         set `project` (or OPENJIUWEN_GCS_PROJECT / GOOGLE_CLOUD_PROJECT)"
                    ),
                )
            })
    }
}

#[async_trait]
impl ObjectStore for GcsObjectStore {
    async fn create_bucket(&self, bucket: &str) -> Result<()> {
        validate_bucket(bucket)?;
        let project = self.project("create")?.to_string();
        let request = HttpRequest::post(format!("{}/{JSON_PATH}/b", self.base))
            .query("project", project)
            .json(&serde_json::json!({ "name": bucket }))?;
        let response = self.send(request).await?;
        // A bucket that already exists is a successful no-op for an idempotent create.
        if response.is_success() || response.status == 409 {
            Ok(())
        } else {
            Err(failure(SERVICE, "create_bucket", &response))
        }
    }

    async fn delete_bucket(&self, bucket: &str) -> Result<()> {
        let url = self.bucket_url(bucket)?;
        let response = self.send(HttpRequest::delete(url)).await?;
        if response.is_success() || response.status == 404 {
            Ok(())
        } else {
            Err(failure(SERVICE, "delete_bucket", &response))
        }
    }

    async fn list_buckets(&self) -> Result<Vec<String>> {
        let project = self.project("list")?.to_string();
        let request =
            HttpRequest::get(format!("{}/{JSON_PATH}/b", self.base)).query("project", project);
        let response = self.send_ok(request, "list_buckets").await?;
        let page = decode::<BucketList>(&response, "a GCS bucket listing")?;
        let mut names: Vec<String> = page
            .items
            .into_iter()
            .map(|bucket| bucket.name)
            .filter(|name| !name.is_empty())
            .collect();
        names.sort();
        names.dedup();
        Ok(names)
    }

    async fn put(&self, bucket: &str, key: &str, data: Bytes) -> Result<ObjectMeta> {
        validate_bucket(bucket)?;
        validate_key(key)?;
        let payload_len = data.len() as u64;
        let request = HttpRequest::post(format!("{}/{UPLOAD_PATH}/b/{bucket}/o", self.base))
            .header("content-type", "application/octet-stream")
            .query("uploadType", "media")
            .query("name", key)
            .bytes(data);
        let response = self.send(request).await?;
        if !response.is_success() {
            return Err(failure(SERVICE, "put", &response));
        }
        // A media upload answers with the object resource. A body-less reply (an emulator, or
        // an intermediary that swallowed it) is tolerated: the key, the payload length and the
        // current instant describe the object just as well.
        let resource: ObjectResource = serde_json::from_slice(&response.body).unwrap_or_default();
        Ok(meta_from_resource(key, &resource, payload_len))
    }

    async fn get(&self, bucket: &str, key: &str) -> Result<Option<Bytes>> {
        let request = HttpRequest::get(self.object_url(bucket, key)?).query("alt", "media");
        let response = self.send(request).await?;
        if response.status == 404 {
            return absent(&response, bucket);
        }
        if !response.is_success() {
            return Err(failure(SERVICE, "get", &response));
        }
        Ok(Some(response.body.clone()))
    }

    async fn delete(&self, bucket: &str, key: &str) -> Result<()> {
        let request = HttpRequest::delete(self.object_url(bucket, key)?);
        let response = self.send(request).await?;
        if response.is_success() || response.status == 404 {
            Ok(())
        } else {
            Err(failure(SERVICE, "delete", &response))
        }
    }

    async fn exists(&self, bucket: &str, key: &str) -> Result<bool> {
        let request = HttpRequest::get(self.object_url(bucket, key)?);
        let response = self.send(request).await?;
        if response.status == 404 {
            return Ok(false);
        }
        if response.is_success() {
            Ok(true)
        } else {
            Err(failure(SERVICE, "exists", &response))
        }
    }

    async fn list(
        &self,
        bucket: &str,
        prefix: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ObjectMeta>> {
        let url = format!("{}/o", self.bucket_url(bucket)?);
        let mut entries: Vec<ObjectMeta> = Vec::new();
        let mut token: Option<String> = None;

        for _page in 0..MAX_PAGES {
            let mut request = HttpRequest::get(url.clone()).query("prefix", prefix);
            if let Some(token) = &token {
                request = request.query("pageToken", token.clone());
            }
            if let Some(limit) = limit {
                let remaining = limit
                    .saturating_sub(entries.len())
                    .clamp(1, MAX_RESULTS_PER_PAGE);
                request = request.query("maxResults", remaining.to_string());
            }

            let response = self.send(request).await?;
            if response.status == 404 {
                // A listing names no object, so a 404 can only be the bucket.
                return Err(Error::not_found(format!("GCS bucket {bucket:?} not found")));
            }
            if !response.is_success() {
                return Err(failure(SERVICE, "list", &response));
            }
            let page = decode::<ObjectList>(&response, "a GCS object listing")?;
            entries.extend(page.items.iter().map(meta_from_listed));
            if limit.is_some_and(|limit| entries.len() >= limit) {
                break;
            }
            match page.next_page_token {
                Some(next) if !next.is_empty() && token.as_deref() != Some(next.as_str()) => {
                    token = Some(next);
                }
                _ => break,
            }
        }

        entries.sort_by(|left, right| left.key.cmp(&right.key));
        if let Some(limit) = limit {
            entries.truncate(limit);
        }
        Ok(entries)
    }
}

/// One object resource of the GCS JSON API (`Object`).
///
/// `size` is a decimal *string* in a GCS reply (the API declares 64-bit integers that way), but
/// a number is accepted too; both are read by [`json_u64`].
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct ObjectResource {
    /// Object name (the store's key).
    name: String,
    /// Payload length, as a string or a number.
    size: Option<Value>,
    /// Last modification time, RFC 3339.
    updated: Option<String>,
}

/// One page of an object listing (`Objects`).
#[derive(Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct ObjectList {
    /// The page's objects.
    items: Vec<ObjectResource>,
    /// Opaque token naming the next page; absent on the last page.
    next_page_token: Option<String>,
}

/// One page of a bucket listing (`Buckets`).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct BucketList {
    /// The page's buckets.
    items: Vec<BucketResource>,
}

/// One bucket resource, reduced to the field a listing needs.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct BucketResource {
    /// Bucket name.
    name: String,
}

/// Builds the metadata of a stored object from a `put` reply.
///
/// The resource is normalized into the [`ObjectMeta`] document shape and decoded by
/// `serde`, which parses the RFC 3339 `updated` field; `fallback_size` (the uploaded payload
/// length) and the current instant stand in for a missing field.
fn meta_from_resource(key: &str, resource: &ObjectResource, fallback_size: u64) -> ObjectMeta {
    let size = resource
        .size
        .as_ref()
        .and_then(json_u64)
        .unwrap_or(fallback_size);
    meta_decode(key, size, resource.updated.as_deref())
}

/// Builds the metadata of an object a listing reported.
///
/// The payload length of a listed object is reported by GCS, so a missing `size` is `0` — the
/// object is still listed.
fn meta_from_listed(resource: &ObjectResource) -> ObjectMeta {
    let size = resource.size.as_ref().and_then(json_u64).unwrap_or(0);
    meta_decode(&resource.name, size, resource.updated.as_deref())
}

/// Decodes `{key, size, updated_at}` as an [`ObjectMeta`], falling back to the current instant
/// for a missing or unparseable timestamp.
fn meta_decode(key: &str, size: u64, updated: Option<&str>) -> ObjectMeta {
    let mut document = serde_json::Map::new();
    document.insert("key".to_string(), Value::String(key.to_string()));
    document.insert("size".to_string(), Value::Number(size.into()));
    if let Some(updated) = updated {
        document.insert("updated_at".to_string(), Value::String(updated.to_string()));
    }
    serde_json::from_value(Value::Object(document)).unwrap_or_else(|_| ObjectMeta {
        key: key.to_string(),
        size,
        updated_at: now_utc(),
    })
}

/// Reads a non-negative integer that GCS may encode as a JSON number or a decimal string.
fn json_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

/// Decodes a JSON reply, naming the URL in the error context.
fn decode<T: serde::de::DeserializeOwned>(response: &HttpResponse, what: &str) -> Result<T> {
    serde_json::from_slice(&response.body).map_err(|error| {
        Error::external(error).with_context(format!("decoding {what} from {}", response.url))
    })
}

/// Maps a 404 reply on an object read to a missing object or a missing bucket.
///
/// GCS answers both with 404; the bucket case is the only one whose message says the bucket
/// does not exist. Anything else keeps the [`ObjectStore`] convention (a missing object is
/// `Ok(None)`), including a bare 404 from an emulator.
fn absent(response: &HttpResponse, bucket: &str) -> Result<Option<Bytes>> {
    if response
        .text()
        .to_ascii_lowercase()
        .contains("bucket does not exist")
    {
        return Err(Error::not_found(format!("GCS bucket {bucket:?} not found")));
    }
    Ok(None)
}

/// Percent-encodes a GCS object name for use as a single URL path segment.
///
/// A GCS object name is one opaque segment, so `/` becomes `%2F` instead of a separator.
/// Unreserved RFC 3986 characters (`A-Z`, `a-z`, `0-9`, `-`, `_`, `.`, `~`) stay as they are,
/// and every other byte — a space, a reserved character, or one byte of a multi-byte UTF-8
/// character — becomes `%XX` with uppercase hexadecimal digits.
fn encode_object_name(name: &str) -> String {
    let mut encoded = String::with_capacity(name.len());
    for byte in name.bytes() {
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

/// Rejects a bucket name that cannot appear in a path segment.
fn validate_bucket(bucket: &str) -> Result<()> {
    if bucket.trim().is_empty() {
        return Err(validation("bucket", "must not be blank"));
    }
    if bucket
        .chars()
        .any(|character| matches!(character, '/' | '?' | '#' | '%') || character.is_control())
    {
        return Err(validation(
            "bucket",
            format!("`{bucket}` contains a character a bucket name cannot carry"),
        ));
    }
    Ok(())
}

/// Rejects an empty object key.
fn validate_key(key: &str) -> Result<()> {
    if key.is_empty() {
        return Err(validation("key", "must not be empty"));
    }
    Ok(())
}

/// Validates the endpoint and returns it without a trailing slash.
///
/// The endpoint must be absolute HTTP(S) with a host; a path is kept, because an emulator or a
/// proxy may live below one. Hand-rolled rather than parsed, since this crate deliberately
/// carries no URL crate.
fn validate_endpoint(endpoint: &str) -> Result<String> {
    let trimmed = endpoint.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(Error::config("GCS endpoint must not be empty"));
    }
    let Some((scheme, rest)) = trimmed.split_once("://") else {
        return Err(Error::config(format!(
            "invalid GCS endpoint {endpoint:?}: expected an absolute http(s) URL"
        )));
    };
    if !matches!(scheme, "http" | "https") {
        return Err(Error::config(format!(
            "unsupported GCS endpoint scheme {scheme:?}; expected `http` or `https`"
        )));
    }
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    if authority.is_empty() {
        return Err(Error::config(format!(
            "GCS endpoint {endpoint:?} has no host"
        )));
    }
    Ok(trimmed.to_string())
}

/// The first environment variable in `names` that is set to a non-empty value.
fn first_env(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
