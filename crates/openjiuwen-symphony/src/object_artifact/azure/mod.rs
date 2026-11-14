//! Azure Blob Storage object backend over the Blob REST API.
//!
//! [`AzureBlobStore`] implements the
//! [`ObjectStore`](openjiuwen_foundation::store::ObjectStore) contract on top of the Azure Blob
//! REST API, so a capability graph can live in an Azure container exactly like it lives in an
//! S3-compatible bucket or a Google Cloud Storage bucket:
//! [`ObjectGraphArtifactStore::azure_blob`](crate::object_artifact::ObjectGraphArtifactStore::azure_blob)
//! and [`ArtifactTransport::AzureBlob`](crate::object_artifact::ArtifactTransport) both build
//! this transport.
//!
//! # Authentication
//!
//! [`AzureBlobAuth`] names the three ways the Blob service accepts a request:
//!
//! - [`AzureBlobAuth::SharedKey`] signs every request with the storage account key (see
//!   [Signing](#signing)).
//! - [`AzureBlobAuth::Sas`] appends a pre-signed shared access signature token to the query
//!   string. The token carries its own account and permissions, so no `Authorization` header is
//!   sent and nothing is signed. A leading `?` in the configured token is stripped.
//! - [`AzureBlobAuth::Anonymous`] sends neither, which is what a container with public access,
//!   or a local emulator that does not verify signatures, needs.
//!
//! An Azure AD client-credential flow is **not** supported:
//! [`AzureBlobStore::from_client_credentials`] exists so that asking for one fails with a typed
//! [`Error::Unsupported`] instead of silently sending anonymous requests. See
//! [`CLIENT_CREDENTIALS_REASON`].
//!
//! # Endpoints
//!
//! `endpoint` is the account endpoint. It defaults to
//! `https://{account}.blob.core.windows.net` when the account is set, and to
//! [`AZURITE_ENDPOINT`] (the well-known local emulator address) when neither is set — that is
//! what a test or an emulator needs, and it is documented rather than implicit. A path in the
//! endpoint is kept as a prefix for every request, which is how a proxy or the Azurite
//! `/{account}` mount is addressed.
//!
//! | Operation | Request |
//! | --- | --- |
//! | `create_bucket` | `PUT {endpoint}/{container}?restype=container` |
//! | `delete_bucket` | `DELETE {endpoint}/{container}?restype=container` |
//! | `list_buckets` | `GET {endpoint}/?comp=list` |
//! | `put` | `PUT {endpoint}/{container}/{blob}` with `x-ms-blob-type: BlockBlob` |
//! | `get` | `GET {endpoint}/{container}/{blob}` |
//! | `exists` | `HEAD {endpoint}/{container}/{blob}` |
//! | `delete` | `DELETE {endpoint}/{container}/{blob}` |
//! | `list` | `GET {endpoint}/{container}?restype=container&comp=list&prefix=…` |
//!
//! Every request carries `x-ms-version: 2021-12-02` and an `x-ms-date` in RFC 1123 form.
//!
//! A blob name is a `/`-separated path — unlike a GCS object name — so the name is
//! percent-encoded segment by segment ([`encode_blob_name`]) and the separators stay. A
//! container name is a single path segment and is validated as one.
//!
//! # Signing
//!
//! [`AzureBlobAuth::SharedKey`] signs with the hand-rolled [`string_to_sign`] +
//! [`hmac_sha256`] pair: the workspace ships no HMAC crate, so the HMAC is built from `sha2`
//! exactly like the S3 Signature Version 4 signer in `openjiuwen-foundation`. The string to
//! sign is the documented Shared Key layout — verb, the empty Content-* / Date / conditional
//! lines, `Content-Length` (empty for `0`), the canonicalized `x-ms-*` headers sorted by name,
//! and the canonicalized resource `/{account}{path}` plus one `\nname:value` line per query
//! parameter sorted by name. The Authorization header is
//! `SharedKey {account}:{base64(HMAC-SHA256(base64_decode(key), string_to_sign))}`.
//!
//! # Timestamps
//!
//! A Blob write answers with `Last-Modified` in RFC 1123 form and `Content-Length`; both are
//! read back into the [`ObjectMeta`] document shape and decoded by [`ObjectMeta`]'s own
//! `serde` implementation, which parses the RFC 3339 timestamp this module derives from the
//! HTTP date. That keeps the module free of a date-time dependency, and a missing or malformed
//! header falls back to the payload length and the current instant. The reply to a block upload
//! has no body, so its `Content-Length: 0` is read as "no reply body" rather than as an empty
//! object.
//!
//! # XML listings
//!
//! Both listing documents are XML. No XML crate exists in the workspace, so the two flat
//! documents are read by a tolerant tag scan ([`tag_texts`]): attributes are ignored,
//! self-closing and unclosed tags are skipped, and malformed input yields whatever was parsed
//! so far instead of panicking.

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use bytes::Bytes;
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, HttpResponse};
use openjiuwen_common::security::SecretString;
use openjiuwen_common::utils::id::now_utc;
use openjiuwen_foundation::store::{ObjectMeta, ObjectStore};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::http::{failure, hmac_sha256};
use crate::error::{unsupported, Error, Result};

mod paths;
mod signing;
#[cfg(test)]
mod tests;
mod xml;

use signing::SigningKey;

/// DNS suffix of the public Blob endpoint, appended to the storage account name.
pub const DEFAULT_ENDPOINT_SUFFIX: &str = "blob.core.windows.net";

/// Endpoint used when neither an account nor an endpoint is configured.
///
/// It is the well-known address of the Azurite local emulator, so a test or a local run needs
/// no configuration at all; every real deployment sets the account (or the endpoint).
pub const AZURITE_ENDPOINT: &str = "http://127.0.0.1:10000";

/// Blob service REST API version sent as `x-ms-version` on every request.
pub const API_VERSION: &str = "2021-12-02";

/// Why [`AzureBlobStore::from_client_credentials`] refuses to mint an Azure AD token.
pub const CLIENT_CREDENTIALS_REASON: &str = "Azure AD client-credential OAuth for Blob Storage \
     is not supported (the identity-endpoint token exchange and its cache are out of scope); \
     configure a storage account shared key or a shared access signature (SAS) token instead";

/// Short service name used in diagnostics.
const SERVICE: &str = "Azure Blob";

/// Maximum number of blobs the service returns in one listing page.
const MAX_RESULTS_PER_PAGE: usize = 5000;

/// Hard cap on the pages one listing call follows, so a misbehaving service cannot make the
/// call loop forever.
const MAX_PAGES: usize = 1000;

/// Timeout used when the configuration does not set one.
const fn default_timeout() -> Duration {
    Duration::from_secs(60)
}

/// How a [`AzureBlobStore`] authenticates against the Blob service.
///
/// [`AzureBlobAuth::Anonymous`] is the default, so a configuration without credentials talks to
/// an emulator (or a public container) instead of failing.
///
/// `Debug` redacts both secrets — the shared key and the SAS token — while `serde` keeps them,
/// so a configuration still round-trips through JSON.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AzureBlobAuth {
    /// Sign every request with the storage account name and key (Shared Key authorization).
    SharedKey {
        /// Storage account the key belongs to.
        account: String,
        /// Base64-encoded storage account key; never printed.
        key: SecretString,
    },
    /// Append a pre-signed shared access signature token to every request.
    Sas {
        /// The SAS token, with or without a leading `?`; never printed.
        token: String,
    },
    /// Send no credentials at all.
    #[default]
    Anonymous,
}

impl AzureBlobAuth {
    /// Builds a [`AzureBlobAuth::SharedKey`] authorization.
    #[must_use]
    pub fn shared_key(account: impl Into<String>, key: impl Into<SecretString>) -> Self {
        Self::SharedKey {
            account: account.into(),
            key: key.into(),
        }
    }

    /// Builds a [`AzureBlobAuth::Sas`] authorization.
    #[must_use]
    pub fn sas(token: impl Into<String>) -> Self {
        Self::Sas {
            token: token.into(),
        }
    }

    /// The authorization name used in diagnostics and in JSON.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::SharedKey { .. } => "shared_key",
            Self::Sas { .. } => "sas",
            Self::Anonymous => "anonymous",
        }
    }

    /// The storage account the authorization names, when it names one.
    ///
    /// A SAS token carries its account inside the token, so only a shared key reports one here.
    #[must_use]
    pub fn account(&self) -> Option<&str> {
        match self {
            Self::SharedKey { account, .. } => Some(account),
            Self::Sas { .. } | Self::Anonymous => None,
        }
    }
}

impl fmt::Debug for AzureBlobAuth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SharedKey { account, .. } => formatter
                .debug_struct("SharedKey")
                .field("account", account)
                .field("key", &"<redacted>")
                .finish(),
            Self::Sas { .. } => formatter
                .debug_struct("Sas")
                .field("token", &"<redacted>")
                .finish(),
            Self::Anonymous => formatter.write_str("Anonymous"),
        }
    }
}

/// Configuration of an [`AzureBlobStore`].
///
/// Every field has a default, so a configuration round-trips through JSON and a partial
/// document keeps the intended values. [`AzureBlobConfig::from_env`] reads the conventional
/// variables, and the `with_*` setters cover code.
///
/// ```
/// use openjiuwen_symphony::object_artifact::AzureBlobConfig;
///
/// let config = AzureBlobConfig::new("demoaccount")
///     .with_endpoint("http://127.0.0.1:10000")
///     .with_shared_key("demoaccount", "a2V5");
///
/// assert_eq!(config.account, "demoaccount");
/// assert_eq!(config.auth.as_str(), "shared_key");
/// assert!(!format!("{config:?}").contains("a2V5"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct AzureBlobConfig {
    /// Storage account name; also the host label of the default endpoint.
    pub account: String,
    /// Account endpoint. Empty means "derive it": `https://{account}.blob.core.windows.net`
    /// when the account is set, [`AZURITE_ENDPOINT`] otherwise.
    pub endpoint: String,
    /// How requests are authenticated.
    pub auth: AzureBlobAuth,
    /// Timeout of a single HTTP request.
    pub timeout: Duration,
}

impl Default for AzureBlobConfig {
    fn default() -> Self {
        Self {
            account: String::new(),
            endpoint: String::new(),
            auth: AzureBlobAuth::Anonymous,
            timeout: default_timeout(),
        }
    }
}

impl AzureBlobConfig {
    /// Creates a configuration for `account` and using every other default.
    #[must_use]
    pub fn new(account: impl Into<String>) -> Self {
        Self {
            account: account.into(),
            ..Self::default()
        }
    }

    /// Sets the storage account name.
    #[must_use]
    pub fn with_account(mut self, account: impl Into<String>) -> Self {
        self.account = account.into();
        self
    }

    /// Sets the account endpoint (an empty value derives it from the account name).
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Sets how every request is authenticated.
    #[must_use]
    pub fn with_auth(mut self, auth: AzureBlobAuth) -> Self {
        self.auth = auth;
        self
    }

    /// Authenticates with a storage account name and key (Shared Key authorization).
    #[must_use]
    pub fn with_shared_key(self, account: impl Into<String>, key: impl Into<SecretString>) -> Self {
        self.with_auth(AzureBlobAuth::shared_key(account, key))
    }

    /// Authenticates with a shared access signature token.
    #[must_use]
    pub fn with_sas_token(self, token: impl Into<String>) -> Self {
        self.with_auth(AzureBlobAuth::sas(token))
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
    /// | account | `OPENJIUWEN_AZURE_STORAGE_ACCOUNT`, `AZURE_STORAGE_ACCOUNT` |
    /// | endpoint | `OPENJIUWEN_AZURE_STORAGE_ENDPOINT`, `AZURE_STORAGE_BLOB_ENDPOINT` |
    /// | shared key | `OPENJIUWEN_AZURE_STORAGE_KEY`, `AZURE_STORAGE_KEY` |
    /// | SAS token | `OPENJIUWEN_AZURE_STORAGE_SAS`, `AZURE_STORAGE_SAS_TOKEN` |
    ///
    /// A configured key wins over a configured SAS token, which wins over anonymous access; the
    /// key's account is the configured account (an empty one is reported when the store is
    /// built). Empty variables count as unset, so this constructor cannot fail.
    #[must_use]
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Some(account) =
            first_env(&["OPENJIUWEN_AZURE_STORAGE_ACCOUNT", "AZURE_STORAGE_ACCOUNT"])
        {
            config.account = account;
        }
        if let Some(endpoint) = first_env(&[
            "OPENJIUWEN_AZURE_STORAGE_ENDPOINT",
            "AZURE_STORAGE_BLOB_ENDPOINT",
        ]) {
            config.endpoint = endpoint;
        }
        if let Some(key) = first_env(&["OPENJIUWEN_AZURE_STORAGE_KEY", "AZURE_STORAGE_KEY"]) {
            config.auth = AzureBlobAuth::shared_key(config.account.clone(), key);
        } else if let Some(token) =
            first_env(&["OPENJIUWEN_AZURE_STORAGE_SAS", "AZURE_STORAGE_SAS_TOKEN"])
        {
            config.auth = AzureBlobAuth::sas(token);
        }
        config
    }
}

/// An [`ObjectStore`](openjiuwen_foundation::store::ObjectStore) backed by Azure Blob Storage.
///
/// The store holds one HTTP client, one validated endpoint and the resolved credentials, so it
/// is cheap to share behind an `Arc` and every method takes `&self`.
///
/// # Errors
///
/// [`AzureBlobStore::new`] validates the endpoint and the credentials. Every operation returns
/// the errors documented on
/// [`ObjectStore`](openjiuwen_foundation::store::ObjectStore), plus [`Error::Validation`] for a
/// blank or malformed container name or blob key, and an [`Error::External`] naming the
/// operation, the status, the URL and a body excerpt when the service rejects a request.
#[derive(Debug, Clone)]
pub struct AzureBlobStore {
    /// HTTP client shared by every request.
    client: HttpClient,
    /// Configuration the store was built from.
    config: AzureBlobConfig,
    /// Validated account endpoint without a trailing slash.
    base: String,
    /// Path component of the endpoint (a proxy prefix or the Azurite account mount).
    path_prefix: String,
    /// Shared Key material, when the store signs its requests.
    signing: Option<SigningKey>,
    /// SAS token, when the store appends one to every query string.
    sas: Option<String>,
}

/// The HTTP verbs this backend uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verb {
    /// `GET`
    Get,
    /// `PUT`
    Put,
    /// `DELETE`
    Delete,
    /// `HEAD`
    Head,
}

impl Verb {
    /// The method name as it appears in the string to sign.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Put => "PUT",
            Self::Delete => "DELETE",
            Self::Head => "HEAD",
        }
    }

    /// Starts a request for this verb.
    fn request(self, url: impl Into<String>) -> HttpRequest {
        match self {
            Self::Get => HttpRequest::get(url),
            Self::Put => HttpRequest::put(url),
            Self::Delete => HttpRequest::delete(url),
            Self::Head => HttpRequest::head(url),
        }
    }
}

/// One Blob request, before the credentials are applied.
///
/// The parts are kept separate because the Shared Key signature covers the verb, the path, the
/// query and the `x-ms-*` headers, and is computed before the URL is assembled.
#[derive(Debug, Clone)]
struct BlobRequest {
    /// HTTP verb.
    verb: Verb,
    /// Path below the endpoint, already percent-encoded.
    path: String,
    /// Query parameters, in insertion order.
    query: Vec<(String, String)>,
    /// Request headers, by lowercase name.
    headers: BTreeMap<String, String>,
    /// Request body, when the verb carries one.
    body: Option<Bytes>,
}

impl BlobRequest {
    /// Starts a request for `verb` on `path`.
    fn new(verb: Verb, path: impl Into<String>) -> Self {
        Self {
            verb,
            path: path.into(),
            query: Vec::new(),
            headers: BTreeMap::new(),
            body: None,
        }
    }

    /// Appends one query parameter.
    fn query(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.query.push((name.into(), value.into()));
        self
    }

    /// Appends one header.
    fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Sets the request body.
    fn body(mut self, body: Bytes) -> Self {
        self.body = Some(body);
        self
    }
}

impl AzureBlobStore {
    /// Builds a store from `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the resolved endpoint is not an absolute HTTP(S) URL,
    /// [`Error::Validation`] naming `account`, `key` or `token` when a credential is malformed
    /// (a shared key must be base64, a SAS token must not be blank), and propagates the HTTP
    /// client's configuration errors.
    pub fn new(config: AzureBlobConfig) -> Result<Self> {
        let (base, path_prefix) = paths::endpoint_parts(&config)?;
        let signing = match &config.auth {
            AzureBlobAuth::SharedKey { account, key } => Some(SigningKey::new(account, key)?),
            AzureBlobAuth::Sas { .. } | AzureBlobAuth::Anonymous => None,
        };
        let sas = match &config.auth {
            AzureBlobAuth::Sas { token } => Some(paths::normalize_sas_token(token)?),
            AzureBlobAuth::SharedKey { .. } | AzureBlobAuth::Anonymous => None,
        };
        let client = HttpClient::new(HttpClientConfig::default().with_timeout(config.timeout))?;
        Ok(Self {
            client,
            config,
            base,
            path_prefix,
            signing,
            sas,
        })
    }

    /// Returns the configuration the store was built from.
    #[must_use]
    pub const fn config(&self) -> &AzureBlobConfig {
        &self.config
    }

    /// Reports that the Azure AD client-credential OAuth flow is not supported.
    ///
    /// Minting a Blob token from a client secret means calling the identity endpoint and
    /// caching the reply, which is outside this crate's scope. The call therefore fails with
    /// [`Error::Unsupported`] instead of falling back to anonymous requests, which would look
    /// like a permission problem on the service side.
    ///
    /// Supply a storage account shared key ([`AzureBlobConfig::with_shared_key`]) or a SAS
    /// token ([`AzureBlobConfig::with_sas_token`]) instead.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`] whose message names the tenant and the client
    /// application and carries [`CLIENT_CREDENTIALS_REASON`]. The client secret is not part of
    /// the message.
    pub fn from_client_credentials(
        _tenant_id: &str,
        client_id: &str,
        _client_secret: &str,
    ) -> Result<Self> {
        Err(unsupported(&format!(
            "{CLIENT_CREDENTIALS_REASON} (refused client `{client_id}`)"
        )))
    }

    /// The path of `container` below the endpoint.
    fn container_path(&self, container: &str) -> Result<String> {
        paths::validate_container(container)?;
        Ok(format!("{}/{container}", self.path_prefix))
    }

    /// The path of `key` inside `container`, with the key percent-encoded per segment.
    fn blob_path(&self, container: &str, key: &str) -> Result<String> {
        paths::validate_key(key)?;
        Ok(format!(
            "{}/{}",
            self.container_path(container)?,
            paths::encode_blob_name(key)
        ))
    }

    /// The path of the account root (`/`, or the endpoint's own path).
    fn root_path(&self) -> String {
        format!("{}/", self.path_prefix)
    }

    /// Applies the credentials and builds the wire request.
    ///
    /// `x-ms-version` and `x-ms-date` are set here so no caller can forget them and so the
    /// signature always covers the values that are actually sent.
    fn build(&self, request: BlobRequest) -> HttpRequest {
        let content_length = request.body.as_ref().map(|body| body.len() as u64);
        let mut request = request;
        request
            .headers
            .insert("x-ms-version".to_string(), API_VERSION.to_string());
        request
            .headers
            .insert("x-ms-date".to_string(), http_date_now());
        if let Some(signer) = &self.signing {
            let string = signing::string_to_sign(
                request.verb,
                content_length,
                &request.headers,
                &request.query,
                &request.path,
                &signer.account,
            );
            let digest = hmac_sha256(&signer.key, string.as_bytes());
            request.headers.insert(
                "authorization".to_string(),
                format!(
                    "{} {}:{}",
                    signing::SCHEME,
                    signer.account,
                    STANDARD.encode(digest)
                ),
            );
        }
        let mut url = format!("{}{}", self.base, request.path);
        if let Some(sas) = &self.sas {
            url.push('?');
            url.push_str(sas);
        }
        let mut built = request
            .verb
            .request(url)
            .timeout(self.config.timeout)
            .queries(request.query)
            .with_headers(request.headers);
        if let Some(body) = request.body {
            built = built.bytes(body);
        }
        built
    }

    /// Sends `request`, leaving the status for the caller to interpret.
    ///
    /// # Errors
    ///
    /// Returns the transport error of the HTTP client.
    async fn send(&self, request: BlobRequest) -> Result<HttpResponse> {
        self.client.execute(self.build(request)).await
    }

    /// Sends `request` and maps any non-2xx status to an error naming `operation`.
    ///
    /// # Errors
    ///
    /// Returns the transport error of the HTTP client, or [`Error::External`] naming the
    /// operation, the status, the URL and a body excerpt for a rejected request.
    async fn send_ok(&self, request: BlobRequest, operation: &str) -> Result<HttpResponse> {
        let response = self.send(request).await?;
        if response.is_success() {
            Ok(response)
        } else {
            Err(failure(SERVICE, operation, &response))
        }
    }
}

#[async_trait]
impl ObjectStore for AzureBlobStore {
    async fn create_bucket(&self, bucket: &str) -> Result<()> {
        let request =
            BlobRequest::new(Verb::Put, self.container_path(bucket)?).query("restype", "container");
        let response = self.send(request).await?;
        // A container that already exists is a successful no-op for an idempotent create.
        if response.is_success() || response.status == 409 {
            Ok(())
        } else {
            Err(failure(SERVICE, "create_bucket", &response))
        }
    }

    async fn delete_bucket(&self, bucket: &str) -> Result<()> {
        let request = BlobRequest::new(Verb::Delete, self.container_path(bucket)?)
            .query("restype", "container");
        let response = self.send(request).await?;
        if response.is_success() || response.status == 404 {
            Ok(())
        } else {
            Err(failure(SERVICE, "delete_bucket", &response))
        }
    }

    async fn list_buckets(&self) -> Result<Vec<String>> {
        let mut names: Vec<String> = Vec::new();
        let mut marker: Option<String> = None;
        for _page in 0..MAX_PAGES {
            let mut request = BlobRequest::new(Verb::Get, self.root_path()).query("comp", "list");
            if let Some(marker) = &marker {
                request = request.query("marker", marker.clone());
            }
            let response = self.send_ok(request, "list_buckets").await?;
            let xml = response.text();
            for block in xml::tag_texts(&xml, "Container") {
                if let Some(name) = xml::first_tag(&block, "Name").filter(|name| !name.is_empty()) {
                    names.push(name);
                }
            }
            match xml::next_marker(&xml, marker.as_deref()) {
                Some(next) => marker = Some(next),
                None => break,
            }
        }
        names.sort();
        names.dedup();
        Ok(names)
    }

    async fn put(&self, bucket: &str, key: &str, data: Bytes) -> Result<ObjectMeta> {
        let length = data.len() as u64;
        let request = BlobRequest::new(Verb::Put, self.blob_path(bucket, key)?)
            .header("x-ms-blob-type", "BlockBlob")
            .header("content-length", length.to_string())
            .body(data);
        let response = self.send_ok(request, "put").await?;
        // A block upload answers with an empty body, so `Content-Length: 0` describes the
        // *reply*, not the stored blob; the uploaded payload length is what the object holds.
        let size = header_value(&response, "content-length")
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|size| *size > 0)
            .unwrap_or(length);
        let updated_at = header_value(&response, "last-modified")
            .as_deref()
            .and_then(parse_http_date);
        Ok(meta(key, size, updated_at))
    }

    async fn get(&self, bucket: &str, key: &str) -> Result<Option<Bytes>> {
        let request = BlobRequest::new(Verb::Get, self.blob_path(bucket, key)?);
        let response = self.send(request).await?;
        if response.status == 404 {
            return Ok(None);
        }
        if !response.is_success() {
            return Err(failure(SERVICE, "get", &response));
        }
        Ok(Some(response.body.clone()))
    }

    async fn delete(&self, bucket: &str, key: &str) -> Result<()> {
        let request = BlobRequest::new(Verb::Delete, self.blob_path(bucket, key)?);
        let response = self.send(request).await?;
        if response.is_success() || response.status == 404 {
            Ok(())
        } else {
            Err(failure(SERVICE, "delete", &response))
        }
    }

    async fn exists(&self, bucket: &str, key: &str) -> Result<bool> {
        let request = BlobRequest::new(Verb::Head, self.blob_path(bucket, key)?);
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
        let path = self.container_path(bucket)?;
        let mut entries: Vec<ObjectMeta> = Vec::new();
        let mut marker: Option<String> = None;

        for _page in 0..MAX_PAGES {
            let page_size = limit.map_or(MAX_RESULTS_PER_PAGE, |limit| {
                limit
                    .saturating_sub(entries.len())
                    .clamp(1, MAX_RESULTS_PER_PAGE)
            });
            let mut request = BlobRequest::new(Verb::Get, path.clone())
                .query("restype", "container")
                .query("comp", "list")
                .query("prefix", prefix)
                .query("maxresults", page_size.to_string());
            if let Some(marker) = &marker {
                request = request.query("marker", marker.clone());
            }
            let response = self.send(request).await?;
            if response.status == 404 {
                // A listing names no blob, so a 404 can only be the container.
                return Err(Error::not_found(format!(
                    "Azure Blob container {bucket:?} not found"
                )));
            }
            if !response.is_success() {
                return Err(failure(SERVICE, "list", &response));
            }
            let xml = response.text();
            append_entries(&mut entries, &xml);
            if limit.is_some_and(|limit| entries.len() >= limit) {
                break;
            }
            match xml::next_marker(&xml, marker.as_deref()) {
                Some(next) => marker = Some(next),
                None => break,
            }
        }

        entries.sort_by(|left, right| left.key.cmp(&right.key));
        if let Some(limit) = limit {
            entries.truncate(limit);
        }
        Ok(entries)
    }
}

/// Appends every `<Blob>` entry of one listing page.
fn append_entries(entries: &mut Vec<ObjectMeta>, page: &str) {
    for block in xml::tag_texts(page, "Blob") {
        let Some(name) = xml::first_tag(&block, "Name").filter(|name| !name.is_empty()) else {
            continue;
        };
        let size = xml::first_tag(&block, "Content-Length")
            .and_then(|size| size.parse::<u64>().ok())
            .unwrap_or(0);
        let updated_at = xml::first_tag(&block, "Last-Modified")
            .as_deref()
            .and_then(parse_http_date);
        entries.push(meta(&name, size, updated_at));
    }
}

/// Builds an [`ObjectMeta`], falling back to the current instant for a missing timestamp.
fn meta(key: &str, size: u64, updated_at: Option<String>) -> ObjectMeta {
    let mut document = serde_json::Map::new();
    document.insert("key".to_string(), Value::String(key.to_string()));
    document.insert("size".to_string(), Value::Number(size.into()));
    if let Some(updated_at) = updated_at {
        document.insert("updated_at".to_string(), Value::String(updated_at));
    }
    serde_json::from_value(Value::Object(document)).unwrap_or_else(|_| ObjectMeta {
        key: key.to_string(),
        size,
        updated_at: now_utc(),
    })
}

/// Parses an HTTP date (`Last-Modified`, RFC 1123 / RFC 2822) into an RFC 3339 string.
///
/// The Blob service answers `Fri, 14 Nov 2026 08:49:00 GMT`; the result is
/// `2026-11-14T08:49:00Z`, which [`ObjectMeta`]'s `serde` implementation parses. A value that
/// already looks like RFC 3339 is passed through, and anything else is `None`, so the caller
/// falls back to the current instant. The parse is hand-rolled to keep the crate free of a
/// date-time dependency; a semantically invalid date (13th month, 30th February) is rejected by
/// the RFC 3339 decoder and ends up as a fallback too.
fn parse_http_date(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    if value.as_bytes().get(4) == Some(&b'-') {
        return Some(value.to_string());
    }
    let mut tokens = value.split_whitespace();
    let mut day = tokens.next()?;
    if day.ends_with(',') {
        day = tokens.next()?;
    }
    let day: u32 = day.parse().ok()?;
    let month = month_number(tokens.next()?)?;
    let year: i32 = tokens.next()?.parse().ok()?;
    let mut time = tokens.next()?.split(':');
    let hour: u32 = time.next()?.parse().ok()?;
    let minute: u32 = time.next()?.parse().ok()?;
    let second: u32 = time.next()?.parse().ok()?;
    if !(1..=31).contains(&day) || hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    Some(format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z"
    ))
}

/// The month number of a three-letter English month name.
fn month_number(name: &str) -> Option<u32> {
    let index = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    .iter()
    .position(|month| month.eq_ignore_ascii_case(name.trim()))?;
    u32::try_from(index).ok().map(|index| index + 1)
}

/// The current instant in RFC 1123 form, as `x-ms-date` requires.
fn http_date_now() -> String {
    now_utc().format("%a, %d %b %Y %H:%M:%S GMT").to_string()
}

/// Reads one response header as a trimmed string.
fn header_value(response: &HttpResponse, name: &str) -> Option<String> {
    response
        .headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().to_string())
}

/// The first environment variable in `names` that is set to a non-empty value.
fn first_env(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    })
}
