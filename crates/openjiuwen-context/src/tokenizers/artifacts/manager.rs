use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest};

use super::cache::{
    artifact_name, cache_dir_from, expand_home, find_artifact, relative_artifact_id, sha256_file,
    sha256_hex, HF_ENDPOINT_ENV,
};
use super::lock::ArtifactLock;
use crate::tokenizers::native::NativeTokenizer;
use crate::tokenizers::{TokenizerEngine, TokenizerSpec};

/// The default `HuggingFace` endpoint, matching Python's mirror default.
pub const DEFAULT_HUGGINGFACE_ENDPOINT: &str = "https://hf-mirror.com";

/// The default `ModelScope` endpoint.
pub const DEFAULT_MODELSCOPE_ENDPOINT: &str = "https://modelscope.cn";

/// The default per-request timeout.
pub const DEFAULT_TOKENIZER_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// The largest artefact body a download accepts, in bytes.
pub const MAX_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;

/// The source value that selects the `HuggingFace` hub.
pub const HUGGINGFACE_SOURCE: &str = "huggingface";

/// The source value that selects `ModelScope`.
pub const MODELSCOPE_SOURCE: &str = "modelscope";

/// The source value reserved for provider-owned endpoints (never a URL).
pub const PROVIDER_OFFICIAL_SOURCE: &str = "provider_official";

/// The branch a `huggingface` download resolves when the spec names no revision.
pub const HUGGINGFACE_DEFAULT_REVISION: &str = "main";

/// The branch a `modelscope` download resolves when the spec names no revision.
pub const MODELSCOPE_DEFAULT_REVISION: &str = "master";

/// Compact code: the spec names no local artefact and no hub source.
pub const LOCAL_ARTIFACT_MISSING_CODE: &str = "local_tokenizer_artifact_missing";

/// Compact code: the expected `tokenizer.json` is absent.
pub const TOKENIZER_JSON_MISSING_CODE: &str = "tokenizer_json_missing";

/// Compact code: the expected `tiktoken.model` vocabulary is absent.
pub const TIKTOKEN_MODEL_MISSING_CODE: &str = "tiktoken_model_missing";

/// Compact code: the spec names no tokenizer id at all.
pub const TOKENIZER_ID_MISSING_CODE: &str = "tokenizer_id_missing";

/// Compact code: downloads are switched off.
pub const REMOTE_DOWNLOAD_DISABLED_CODE: &str = "remote_download_disabled";

/// Compact code: the hub could not be reached or timed out.
pub const TOKENIZER_NETWORK_UNAVAILABLE_CODE: &str = "tokenizer_network_unavailable";

/// Compact code: offline mode and the artefact is not cached.
pub const TOKENIZER_NOT_IN_LOCAL_CACHE_CODE: &str = "tokenizer_not_in_local_cache";

/// Compact code: the revision (or repository) does not exist.
pub const TOKENIZER_REVISION_UNAVAILABLE_CODE: &str = "tokenizer_revision_unavailable";

/// Compact code: the download failed for another reason.
pub const TOKENIZER_DOWNLOAD_FAILED_CODE: &str = "tokenizer_download_failed";

/// Compact code: the artefact bytes do not match the declared SHA-256.
pub const TOKENIZER_SHA256_MISMATCH_CODE: &str = "tokenizer_sha256_mismatch";

/// Compact code: the tokenizer id is an arbitrary URL.
pub const REMOTE_URL_NOT_ALLOWED_CODE: &str = "remote_url_not_allowed";

/// Compact code: a provider-official tokenizer needs an adapter.
pub const PROVIDER_OFFICIAL_ADAPTER_NOT_CONFIGURED_CODE: &str =
    "provider_official_adapter_not_configured";

/// Compact code: the artefact declares a model type this crate does not decode.
///
/// The code is published through [`ARTIFACT_ERROR_CODES`]; the constant itself stays crate-visible because this module's public surface is the re-export list in [`super`], which a downstream crate can extend without changing the manager.
pub(crate) const TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE: &str = "tokenizer_model_type_unsupported";

/// The vendor named in a `provider_official` message when nothing identifies one.
const UNNAMED_VENDOR: &str = "provider-official";

/// Every compact code the manager can report.
pub const ARTIFACT_ERROR_CODES: &[&str] = &[
    LOCAL_ARTIFACT_MISSING_CODE,
    TOKENIZER_JSON_MISSING_CODE,
    TIKTOKEN_MODEL_MISSING_CODE,
    TOKENIZER_ID_MISSING_CODE,
    REMOTE_DOWNLOAD_DISABLED_CODE,
    TOKENIZER_NETWORK_UNAVAILABLE_CODE,
    TOKENIZER_NOT_IN_LOCAL_CACHE_CODE,
    TOKENIZER_REVISION_UNAVAILABLE_CODE,
    TOKENIZER_DOWNLOAD_FAILED_CODE,
    TOKENIZER_SHA256_MISMATCH_CODE,
    REMOTE_URL_NOT_ALLOWED_CODE,
    PROVIDER_OFFICIAL_ADAPTER_NOT_CONFIGURED_CODE,
    TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE,
];

/// The hub family a spec resolves through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HubSource {
    /// A `huggingface` repository.
    HuggingFace,
    /// A `modelscope` model.
    ModelScope,
    /// A provider-owned endpoint, which needs an adapter.
    ProviderOfficial,
    /// A local artefact only.
    Local,
}

/// Resolves, validates and caches native tokenizer artefacts.
///
/// ```
/// use openjiuwen_context::tokenizers::artifacts::TokenizerArtifactManager;
///
/// // Construction is network-free and downloads stay off by default.
/// let manager = TokenizerArtifactManager::new();
/// assert!(!manager.enable_download());
/// assert!(!manager.offline());
/// assert!(manager.last_error().is_none());
/// ```
#[derive(Debug)]
pub struct TokenizerArtifactManager {
    /// Directory holding cached artefacts and download locks.
    cache_dir: PathBuf,
    /// Whether a resolve may download a missing artefact.
    enable_download: bool,
    /// Whether the manager is restricted to the local cache.
    offline: bool,
    /// Explicit hub endpoint, else `HF_ENDPOINT`; `None` means the defaults.
    endpoint: Option<String>,
    /// Per-request timeout.
    request_timeout: Duration,
    /// Compact code of the last failed resolution.
    last_error: Mutex<Option<String>>,
}

impl Default for TokenizerArtifactManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for TokenizerArtifactManager {
    fn clone(&self) -> Self {
        Self {
            cache_dir: self.cache_dir.clone(),
            enable_download: self.enable_download,
            offline: self.offline,
            endpoint: self.endpoint.clone(),
            request_timeout: self.request_timeout,
            last_error: Mutex::new(self.last_error()),
        }
    }
}

impl TokenizerArtifactManager {
    /// Creates a manager with downloads off and the default cache directory.
    ///
    /// The cache directory comes from
    /// [`OPENJIUWEN_TOKENIZER_CACHE_DIR`](super::cache::CACHE_DIR_ENV) or `HOME`; the
    /// endpoint comes from `HF_ENDPOINT` when that is set. Neither step touches
    /// the network.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache_dir: cache_dir_from(None),
            enable_download: false,
            offline: false,
            endpoint: configured_endpoint(),
            request_timeout: DEFAULT_TOKENIZER_REQUEST_TIMEOUT,
            last_error: Mutex::new(None),
        }
    }

    /// Sets the cache directory, overriding the environment.
    ///
    /// A leading `~` is expanded against `HOME`; an empty path is ignored.
    #[must_use]
    pub fn with_cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        let path = cache_dir.into();
        if !path.as_os_str().is_empty() {
            self.cache_dir = expand_home(path);
        }
        self
    }

    /// Allows (or forbids) downloading a missing artefact.
    #[must_use]
    pub fn with_enable_download(mut self, enable_download: bool) -> Self {
        self.enable_download = enable_download;
        self
    }

    /// Restricts the manager to the local cache, even when downloads are on.
    #[must_use]
    pub fn with_offline(mut self, offline: bool) -> Self {
        self.offline = offline;
        self
    }

    /// Sets the hub endpoint used instead of the `huggingface`/`modelscope`
    /// defaults; the value is trimmed and loses trailing slashes.
    ///
    /// An empty value restores the defaults.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        let endpoint = endpoint.into();
        let trimmed = endpoint.trim().trim_end_matches('/');
        self.endpoint = if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_owned())
        };
        self
    }

    /// Sets the per-request timeout. A zero duration restores the default.
    #[must_use]
    pub fn with_request_timeout(mut self, request_timeout: Duration) -> Self {
        self.request_timeout = if request_timeout.is_zero() {
            DEFAULT_TOKENIZER_REQUEST_TIMEOUT
        } else {
            request_timeout
        };
        self
    }

    /// The cache directory artefacts and locks live in.
    #[must_use]
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Whether a resolve may download.
    #[must_use]
    pub fn enable_download(&self) -> bool {
        self.enable_download
    }

    /// Whether the manager is restricted to the local cache.
    #[must_use]
    pub fn offline(&self) -> bool {
        self.offline
    }

    /// The configured `huggingface` endpoint.
    #[must_use]
    pub fn endpoint(&self) -> &str {
        self.endpoint
            .as_deref()
            .unwrap_or(DEFAULT_HUGGINGFACE_ENDPOINT)
    }

    /// The configured `modelscope` endpoint.
    #[must_use]
    pub fn modelscope_endpoint(&self) -> &str {
        self.endpoint
            .as_deref()
            .unwrap_or(DEFAULT_MODELSCOPE_ENDPOINT)
    }

    /// The per-request timeout.
    #[must_use]
    pub fn request_timeout(&self) -> Duration {
        self.request_timeout
    }

    /// The reason of the last failed resolution, if the last one failed.
    ///
    /// The value starts with one of [`ARTIFACT_ERROR_CODES`]: a typed failure keeps its whole message, so a `provider_official` spec reports the vendor and tokenizer id it needs an adapter for, while a failure the manager decides itself (a disabled download, a missing local artefact) is just the code.
    #[must_use]
    pub fn last_error(&self) -> Option<String> {
        self.slot().clone()
    }

    /// The cache path a download of `spec` writes, when it has a hub source.
    ///
    /// The path is computed, never created, so a caller can assert where an
    /// artefact will land.
    #[must_use]
    pub fn cache_target(&self, spec: &TokenizerSpec) -> Option<PathBuf> {
        let id = relative_artifact_id(spec.tokenizer_id.as_deref()?)?;
        let revision = revision_of(spec, source_of(spec));
        let root = match source_of(spec) {
            HubSource::HuggingFace => self.cache_dir.join(HUGGINGFACE_SOURCE),
            HubSource::ModelScope => self.cache_dir.join("models"),
            HubSource::ProviderOfficial | HubSource::Local => return None,
        };
        Some(
            root.join(id)
                .join(revision)
                .join(artifact_name(spec.engine)),
        )
    }

    /// Resolves `spec` to a validated local artefact, downloading when allowed.
    ///
    /// The order matches Python's manager: an `artifact_path` (file or directory), then a `tokenizer_id` that exists on disk, then validation, then the hub source; `://` ids are refused before any source is considered.
    /// Failures return `None` and record the reason read back through [`TokenizerArtifactManager::last_error`], and the call never panics.
    /// A `provider_official` spec names the vendor and tokenizer id it could not resolve, and a downloaded artefact this crate cannot decode records `tokenizer_model_type_unsupported` with the declared model type, so neither failure looks like a cache miss.
    /// Unlike Python this method is `async`, because a download uses the workspace HTTP client; it performs no I/O at all when downloads are off.
    pub async fn resolve(&self, spec: &TokenizerSpec) -> Option<PathBuf> {
        self.clear_error();
        if let Some(path) = Self::local_artifact(spec) {
            return self.validate_async(path, spec).await;
        }
        if let Some(code) = declared_artifact_missing(spec) {
            return self.fail(code);
        }
        self.checked_id(spec)?;
        match source_of(spec) {
            HubSource::HuggingFace | HubSource::ModelScope => {
                self.resolve_hub(spec, source_of(spec)).await
            }
            HubSource::ProviderOfficial => self.fail_error(&provider_official_error(spec)),
            HubSource::Local => self.fail(LOCAL_ARTIFACT_MISSING_CODE),
        }
    }

    /// Resolves `spec` against the local cache only, never the network.
    ///
    /// This is the rung the synchronous tokenizer selector uses: a cached (or explicitly configured) artefact is validated and returned, and anything else records [`REMOTE_DOWNLOAD_DISABLED_CODE`] when downloads are off, or [`TOKENIZER_NOT_IN_LOCAL_CACHE_CODE`] when they are enabled but a synchronous caller cannot wait for one.
    /// A `provider_official` spec records the adapter message naming its vendor and tokenizer id.
    #[must_use]
    pub fn resolve_cached(&self, spec: &TokenizerSpec) -> Option<PathBuf> {
        self.clear_error();
        if let Some(path) = Self::local_artifact(spec) {
            return self.validate_cached(&path, spec);
        }
        if let Some(code) = declared_artifact_missing(spec) {
            return self.fail(code);
        }
        self.checked_id(spec)?;
        let cached = match source_of(spec) {
            HubSource::HuggingFace => self.hub_cached(spec),
            HubSource::ModelScope => self
                .modelscope_cached(spec)
                .or_else(|| self.hub_cached(spec)),
            HubSource::ProviderOfficial => return self.fail_error(&provider_official_error(spec)),
            HubSource::Local => return self.fail(LOCAL_ARTIFACT_MISSING_CODE),
        };
        if let Some(path) = cached {
            return self.validate_cached(&path, spec);
        }
        if self.enable_download && !self.offline {
            return self.fail(TOKENIZER_NOT_IN_LOCAL_CACHE_CODE);
        }
        self.fail(REMOTE_DOWNLOAD_DISABLED_CODE)
    }

    /// Downloads `spec` into the cache, returning the validated artefact path.
    ///
    /// A local artefact or a valid cache entry is returned without a request.
    /// Concurrent downloads of the same spec serialize on an advisory lock file, and the waiter reuses the artefact the winner wrote instead of fetching again.
    /// The body is written to a temporary file in the target directory and renamed into place, so a reader never sees a partial artefact and a failed download leaves nothing behind.
    ///
    /// # Errors
    ///
    /// - [`Error::Config`] when downloads are disabled or the manager is offline (message starts with the compact code);
    /// - [`Error::Unsupported`] for a `provider_official` source, naming the vendor and the tokenizer id, and for a well-formed body that declares a model type this crate does not decode (`tokenizer_model_type_unsupported`, published through [`ARTIFACT_ERROR_CODES`]), which is never cached;
    /// - [`Error::Validation`] when the id, the URL, or the body is unusable, including a SHA-256 mismatch;
    /// - [`Error::Timeout`] / [`Error::Transport`] when the hub is unreachable;
    /// - [`Error::NotFound`] when the repository, revision or file is missing.
    pub async fn download(&self, spec: &TokenizerSpec) -> Result<PathBuf> {
        self.clear_error();
        match self.download_inner(spec).await {
            Ok(path) => Ok(path),
            Err(error) => Err(self.record(error)),
        }
    }

    /// The download body, with every error mapped through [`Self::record`] by
    /// the caller.
    async fn download_inner(&self, spec: &TokenizerSpec) -> Result<PathBuf> {
        if let Some(path) = Self::local_artifact(spec) {
            return Ok(path);
        }
        if !self.enable_download {
            return Err(Error::Config(format!(
                "{REMOTE_DOWNLOAD_DISABLED_CODE}: tokenizer downloads are disabled; enable them on the artifact manager"
            )));
        }
        if self.offline {
            return Err(Error::Config(format!(
                "{TOKENIZER_NOT_IN_LOCAL_CACHE_CODE}: the artifact manager is offline"
            )));
        }
        let source = Self::download_source(spec)?;
        let url = self.artifact_url(spec, source)?;
        let target = self.cache_target(spec).ok_or_else(|| {
            Error::Validation(format!(
                "{TOKENIZER_DOWNLOAD_FAILED_CODE}: {PROVIDER_OFFICIAL_SOURCE} and local tokenizers have no download path"
            ))
        })?;
        if let Some(path) = Self::cached_target(&target, spec) {
            return Ok(path);
        }
        self.create_cache_dir(&target).await?;
        let _lock = ArtifactLock::acquire(self.lock_path(spec)).await?;
        if let Some(path) = Self::cached_target(&target, spec) {
            return Ok(path);
        }
        let bytes = self.fetch(&url).await?;
        // Validation runs before anything is written, so a body this crate
        // cannot decode never enters the cache and is never handed back by
        // `resolve_cached` for the selector to downgrade silently.
        validate_body(spec, &bytes)?;
        verify_digest(spec, &bytes)?;
        self.write_atomically(&target, &bytes).await?;
        Ok(target)
    }

    /// Creates the cache directory and the artefact's parent directory.
    async fn create_cache_dir(&self, target: &Path) -> Result<()> {
        tokio::fs::create_dir_all(&self.cache_dir)
            .await
            .map_err(|error| Error::external(error).with_context(TOKENIZER_DOWNLOAD_FAILED_CODE))?;
        let Some(parent) = target.parent() else {
            return Ok(());
        };
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|error| Error::external(error).with_context(TOKENIZER_DOWNLOAD_FAILED_CODE))
    }

    /// Resolves a hub source that may need a download.
    async fn resolve_hub(&self, spec: &TokenizerSpec, source: HubSource) -> Option<PathBuf> {
        let cached = match source {
            HubSource::HuggingFace => self.hub_cached(spec),
            _ => self
                .modelscope_cached(spec)
                .or_else(|| self.hub_cached(spec)),
        };
        if let Some(path) = cached {
            return self.validate_async(path, spec).await;
        }
        if !self.enable_download {
            return self.fail(REMOTE_DOWNLOAD_DISABLED_CODE);
        }
        if self.offline {
            return self.fail(TOKENIZER_NOT_IN_LOCAL_CACHE_CODE);
        }
        match self.download(spec).await {
            Ok(path) => self.validate_async(path, spec).await,
            Err(error) => self.fail_error(&error),
        }
    }

    /// The hub source a download may use.
    fn download_source(spec: &TokenizerSpec) -> Result<HubSource> {
        match source_of(spec) {
            source @ (HubSource::HuggingFace | HubSource::ModelScope) => Ok(source),
            HubSource::ProviderOfficial => Err(provider_official_error(spec)),
            HubSource::Local => Err(Error::Validation(format!(
                "{LOCAL_ARTIFACT_MISSING_CODE}: a local tokenizer spec has nothing to download"
            ))),
        }
    }

    /// The absolute URL a download fetches.
    fn artifact_url(&self, spec: &TokenizerSpec, source: HubSource) -> Result<String> {
        let raw = spec.tokenizer_id.as_deref().unwrap_or_default().trim();
        let Some(id) = relative_artifact_id(raw) else {
            return Err(Error::Validation(format!(
                "{REMOTE_URL_NOT_ALLOWED_CODE}: tokenizer id {raw:?} is not a repository-relative id"
            )));
        };
        let id = id.to_string_lossy().replace('\\', "/");
        let artifact = artifact_name(spec.engine);
        let revision = revision_of(spec, source);
        Ok(match source {
            HubSource::HuggingFace => {
                format!("{}/{id}/resolve/{revision}/{artifact}", self.endpoint())
            }
            _ => format!(
                "{}/models/{id}/resolve/{revision}/{artifact}",
                self.modelscope_endpoint()
            ),
        })
    }

    /// Fetches one URL with a bounded timeout and no retries.
    async fn fetch(&self, url: &str) -> Result<Vec<u8>> {
        let config = HttpClientConfig::default()
            .with_timeout(self.request_timeout)
            .with_max_retries(0)
            .with_max_response_bytes(MAX_ARTIFACT_BYTES);
        let client = HttpClient::new(config)
            .map_err(|error| Error::external(error).with_context(TOKENIZER_DOWNLOAD_FAILED_CODE))?;
        let response = client
            .execute(HttpRequest::get(url).timeout(self.request_timeout))
            .await
            .map_err(|error| transport_error(error, url))?;
        if !response.is_success() {
            let status = response.status;
            let code = if matches!(status, 401 | 403 | 404) {
                TOKENIZER_REVISION_UNAVAILABLE_CODE
            } else {
                TOKENIZER_DOWNLOAD_FAILED_CODE
            };
            return Err(match status {
                401 | 403 | 404 => Error::NotFound(format!("{code}: HTTP {status} for {url}")),
                _ => Error::Transport(format!("{code}: HTTP {status} for {url}")),
            });
        }
        Ok(response.bytes().to_vec())
    }

    /// Writes `bytes` to a temporary file beside `target` and renames it.
    async fn write_atomically(&self, target: &Path, bytes: &[u8]) -> Result<()> {
        let Some(parent) = target.parent() else {
            return Err(Error::Validation(format!(
                "{TOKENIZER_DOWNLOAD_FAILED_CODE}: cache path {} has no parent",
                target.display()
            )));
        };
        let temp = parent.join(format!(".openjiuwen-{}.tmp", uuid::Uuid::new_v4()));
        if let Err(error) = tokio::fs::write(&temp, bytes).await {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(Error::external(error).with_context(TOKENIZER_DOWNLOAD_FAILED_CODE));
        }
        if let Err(error) = tokio::fs::rename(&temp, target).await {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(Error::external(error).with_context(TOKENIZER_DOWNLOAD_FAILED_CODE));
        }
        Ok(())
    }

    /// The lock file guarding one spec's download.
    fn lock_path(&self, spec: &TokenizerSpec) -> PathBuf {
        let key = sha256_hex(cache_key(spec).as_bytes());
        self.cache_dir.join(format!("{key}.lock"))
    }

    /// A cache entry that already validates, without touching the file twice.
    ///
    /// A file that cannot be read is not a silent "no cache": it is logged with
    /// its path and the IO error and treated as a miss, so the caller downloads
    /// the artefact again instead of reusing bytes nobody could digest.
    fn cached_target(target: &Path, spec: &TokenizerSpec) -> Option<PathBuf> {
        if !target.is_file() {
            return None;
        }
        let Some(expected) = expected_digest(spec) else {
            return Some(target.to_path_buf());
        };
        match sha256_file(target) {
            Ok(actual) => (actual == expected).then(|| target.to_path_buf()),
            Err(error) => {
                tracing::warn!(
                    path = %target.display(),
                    %error,
                    "cached tokenizer artefact could not be read; downloading it again"
                );
                None
            }
        }
    }

    /// The cached artefact of the exact revision a download would write.
    fn hub_cached(&self, spec: &TokenizerSpec) -> Option<PathBuf> {
        let target = self.cache_target(spec)?;
        target.is_file().then_some(target)
    }

    /// The `modelscope` offline snapshot layouts, probed before any network.
    ///
    /// Python has shipped `<cache>/<model>`, `<cache>/models/<model>` and
    /// `<cache>/hub/models/<model>`; all three are searched, and the search
    /// stays inside directories derived from the id, so an unrelated cached
    /// tokenizer can never be selected.
    fn modelscope_cached(&self, spec: &TokenizerSpec) -> Option<PathBuf> {
        let id = relative_artifact_id(spec.tokenizer_id.as_deref()?)?;
        let layouts = [
            self.cache_dir.join(&id),
            self.cache_dir.join("models").join(&id),
            self.cache_dir.join("hub").join("models").join(&id),
        ];
        layouts
            .iter()
            .find_map(|root| find_artifact(root, spec.engine).filter(|path| path.is_file()))
    }

    /// The artefact an explicit path or an on-disk `tokenizer_id` names.
    fn local_artifact(spec: &TokenizerSpec) -> Option<PathBuf> {
        let explicit = spec
            .artifact_path
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        if let Some(path) = explicit {
            return find_artifact(&expand_home(path), spec.engine).filter(|path| path.is_file());
        }
        let id = spec.tokenizer_id.as_deref().unwrap_or_default().trim();
        if id.is_empty() {
            return None;
        }
        let candidate = expand_home(id);
        if !candidate.exists() {
            return None;
        }
        find_artifact(&candidate, spec.engine).filter(|path| path.is_file())
    }

    /// The tokenizer id, or a recorded code when it cannot be used.
    fn checked_id(&self, spec: &TokenizerSpec) -> Option<String> {
        let id = spec.tokenizer_id.as_deref().unwrap_or_default().trim();
        if id.is_empty() {
            self.set_error(TOKENIZER_ID_MISSING_CODE);
            return None;
        }
        if id.contains("://") {
            self.set_error(REMOTE_URL_NOT_ALLOWED_CODE);
            return None;
        }
        Some(id.to_owned())
    }

    /// Validates an artefact on the synchronous seam.
    fn validate_cached(&self, path: &Path, spec: &TokenizerSpec) -> Option<PathBuf> {
        match validate_file(path, spec) {
            Ok(path) => {
                self.clear_error();
                Some(path)
            }
            Err(code) => self.fail(code),
        }
    }

    /// Validates an artefact off the async task.
    async fn validate_async(&self, path: PathBuf, spec: &TokenizerSpec) -> Option<PathBuf> {
        let expected = expected_digest(spec);
        let missing = missing_artifact_code(spec.engine);
        let validated = tokio::task::spawn_blocking(move || {
            validate_file_with(path, expected.as_deref(), missing)
        })
        .await;
        match validated {
            Ok(Ok(path)) => {
                self.clear_error();
                Some(path)
            }
            Ok(Err(code)) => self.fail(code),
            Err(error) => {
                tracing::warn!(%error, "tokenizer artefact validation task failed");
                self.fail_error(&Error::Validation(format!(
                    "{TOKENIZER_DOWNLOAD_FAILED_CODE}: tokenizer artefact validation task failed: {error}"
                )))
            }
        }
    }

    /// The error slot; a poisoned lock still yields the last value.
    fn slot(&self) -> std::sync::MutexGuard<'_, Option<String>> {
        self.last_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Forgets the previous failure.
    fn clear_error(&self) {
        *self.slot() = None;
    }

    /// Records a compact failure code.
    fn set_error(&self, code: &str) {
        *self.slot() = Some(code.to_owned());
    }

    /// Records the whole message of `error`, code included.
    ///
    /// A typed failure keeps its detail — the vendor a `provider_official` spec
    /// needs, the model type a body declared, the status a hub answered — so
    /// [`TokenizerArtifactManager::last_error`] explains the downgrade instead of
    /// only labelling it.
    fn set_message(&self, error: &Error) {
        *self.slot() = Some(error.message().into_owned());
    }

    /// Records a compact failure code and returns `None`.
    fn fail<T>(&self, code: &str) -> Option<T> {
        self.set_error(code);
        None
    }

    /// Records the whole message of `error` and returns `None`.
    fn fail_error<T>(&self, error: &Error) -> Option<T> {
        self.set_message(error);
        None
    }

    /// Records the message of `error` and returns the error unchanged.
    fn record(&self, error: Error) -> Error {
        self.set_message(&error);
        error
    }
}

/// The endpoint configured through `HF_ENDPOINT`, when it is set.
fn configured_endpoint() -> Option<String> {
    let endpoint = std::env::var(HF_ENDPOINT_ENV).ok()?;
    let trimmed = endpoint.trim().trim_end_matches('/');
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

/// The hub family `spec` resolves through.
fn source_of(spec: &TokenizerSpec) -> HubSource {
    match spec
        .source
        .as_deref()
        .unwrap_or_default()
        .trim()
        .to_lowercase()
        .as_str()
    {
        HUGGINGFACE_SOURCE => HubSource::HuggingFace,
        MODELSCOPE_SOURCE => HubSource::ModelScope,
        PROVIDER_OFFICIAL_SOURCE => HubSource::ProviderOfficial,
        _ => HubSource::Local,
    }
}

/// The revision a download or cache path uses.
fn revision_of(spec: &TokenizerSpec, source: HubSource) -> String {
    let revision = spec.revision.as_deref().unwrap_or_default().trim();
    if !revision.is_empty() {
        return revision.to_owned();
    }
    match source {
        HubSource::ModelScope => MODELSCOPE_DEFAULT_REVISION.to_owned(),
        _ => HUGGINGFACE_DEFAULT_REVISION.to_owned(),
    }
}

/// The declared digest, normalized to lowercase hex.
fn expected_digest(spec: &TokenizerSpec) -> Option<String> {
    spec.sha256
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
}

/// The lock-key material of one spec, joined with NUL like Python's.
fn cache_key(spec: &TokenizerSpec) -> String {
    let engine = match spec.engine {
        TokenizerEngine::Tiktoken => "tiktoken",
        TokenizerEngine::Native => "tokenizers",
        TokenizerEngine::Auto => "auto",
    };
    [
        spec.source.as_deref().unwrap_or_default(),
        spec.tokenizer_id.as_deref().unwrap_or_default(),
        spec.revision.as_deref().unwrap_or_default(),
        spec.sha256.as_deref().unwrap_or_default(),
        engine,
    ]
    .join("\u{0}")
}

/// The compact code reported when the expected artefact is absent.
fn missing_artifact_code(engine: TokenizerEngine) -> &'static str {
    if engine == TokenizerEngine::Tiktoken {
        TIKTOKEN_MODEL_MISSING_CODE
    } else {
        TOKENIZER_JSON_MISSING_CODE
    }
}

/// The code for a declared `artifact_path` that holds no usable artefact.
///
/// A spec that names a path is reporting a local installation, so a missing
/// file is worth distinguishing from "no local artefact was configured".
fn declared_artifact_missing(spec: &TokenizerSpec) -> Option<&'static str> {
    let declared = spec
        .artifact_path
        .as_deref()
        .map(str::trim)
        .is_some_and(|value| !value.is_empty());
    declared.then(|| missing_artifact_code(spec.engine))
}

/// Validates existence and, when declared, the digest of an artefact.
fn validate_file(path: &Path, spec: &TokenizerSpec) -> std::result::Result<PathBuf, &'static str> {
    validate_file_with(
        path.to_path_buf(),
        expected_digest(spec).as_deref(),
        missing_artifact_code(spec.engine),
    )
}

/// Validates one artefact path against an already-normalized digest.
fn validate_file_with(
    path: PathBuf,
    expected: Option<&str>,
    missing: &'static str,
) -> std::result::Result<PathBuf, &'static str> {
    if !path.is_file() {
        return Err(missing);
    }
    if let Some(expected) = expected {
        let actual = sha256_file(&path).map_err(|_| TOKENIZER_DOWNLOAD_FAILED_CODE)?;
        if actual != expected {
            return Err(TOKENIZER_SHA256_MISMATCH_CODE);
        }
    }
    Ok(path)
}

/// Refuses a body that cannot be the artefact its name promises.
///
/// A well-formed document of a model type this crate does not decode is a typed [`Error::Unsupported`] carrying [`TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE`] and the declared type, so the caller learns why counting falls back to `tiktoken` or the estimate instead of getting a silent downgrade.
/// The bytes are not cached: an artefact this loader cannot decode would be handed straight back by [`TokenizerArtifactManager::resolve_cached`] for the selector to downgrade again, so the precedence is reject-before-write and a later loader that decodes the type downloads it afresh.
/// Malformed text is rejected with [`Error::Validation`] and [`TOKENIZER_DOWNLOAD_FAILED_CODE`], and the SHA-256 check is unchanged.
fn validate_body(spec: &TokenizerSpec, bytes: &[u8]) -> Result<()> {
    let text = std::str::from_utf8(bytes).map_err(|error| {
        Error::Validation(format!(
            "{TOKENIZER_DOWNLOAD_FAILED_CODE}: artefact body is not UTF-8: {error}"
        ))
    })?;
    let decoded = match spec.engine {
        TokenizerEngine::Tiktoken => NativeTokenizer::from_tiktoken_model(text).map(|_| ()),
        TokenizerEngine::Auto | TokenizerEngine::Native => {
            NativeTokenizer::from_json_str(text).map(|_| ())
        }
    };
    match decoded {
        Ok(()) => Ok(()),
        Err(Error::Unsupported(message)) => {
            let id = spec
                .tokenizer_id
                .as_deref()
                .map(str::trim)
                .unwrap_or_default();
            let artifact = if id.is_empty() {
                format!("the {} artefact", artifact_name(spec.engine))
            } else {
                format!("the tokenizer artefact {id:?}")
            };
            Err(Error::Unsupported(format!(
                "{TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE}: {artifact} declares a model type this loader does not decode: {message}"
            )))
        }
        Err(error) => Err(Error::Validation(format!(
            "{TOKENIZER_DOWNLOAD_FAILED_CODE}: artefact body is not a usable tokenizer: {error}"
        ))),
    }
}

/// The typed refusal of a `provider_official` spec, naming vendor and id.
///
/// No provider-hosted tokenizer client ships in this workspace, so the source stays unimplemented; the message still identifies what was asked for, because "provider official" alone names neither the vendor nor the artefact.
/// The vendor comes from the spec's `provider` field, then the `vendor/name` prefix of the tokenizer id, then `provider-official`; an unrecognised provider is reported verbatim instead of being mapped onto a vendor this crate happens to know.
fn provider_official_error(spec: &TokenizerSpec) -> Error {
    let vendor = provider_official_vendor(spec);
    let id = spec
        .tokenizer_id
        .as_deref()
        .map(str::trim)
        .unwrap_or_default();
    let id = if id.is_empty() { "<unnamed>" } else { id };
    Error::Unsupported(format!(
        "{PROVIDER_OFFICIAL_ADAPTER_NOT_CONFIGURED_CODE}: {vendor}:{id} needs a {vendor} tokenizer adapter (no provider-hosted tokenizer client is shipped)"
    ))
}

/// The vendor a `provider_official` spec belongs to.
fn provider_official_vendor(spec: &TokenizerSpec) -> String {
    let provider = spec.provider.trim();
    if !provider.is_empty() {
        return provider.to_owned();
    }
    let vendor = spec
        .tokenizer_id
        .as_deref()
        .map(str::trim)
        .and_then(|id| id.split_once('/'));
    match vendor {
        Some((vendor, _)) if !vendor.trim().is_empty() => vendor.trim().to_owned(),
        _ => UNNAMED_VENDOR.to_owned(),
    }
}

/// Compares the downloaded bytes with the declared digest.
fn verify_digest(spec: &TokenizerSpec, bytes: &[u8]) -> Result<()> {
    let Some(expected) = expected_digest(spec) else {
        return Ok(());
    };
    if sha256_hex(bytes) == expected {
        return Ok(());
    }
    Err(Error::Validation(format!(
        "{TOKENIZER_SHA256_MISMATCH_CODE}: downloaded artefact does not match sha256 {expected}"
    )))
}

/// Maps a transport failure onto the network-unavailable code.
fn transport_error(error: Error, url: &str) -> Error {
    if error.kind() == ErrorKind::Timeout {
        Error::Timeout(format!(
            "{TOKENIZER_NETWORK_UNAVAILABLE_CODE}: {url} timed out: {error}"
        ))
    } else {
        Error::Transport(format!(
            "{TOKENIZER_NETWORK_UNAVAILABLE_CODE}: {url} is unreachable: {error}"
        ))
    }
}

/// The compact code an error carries, else the download-failed default.
///
/// [`TokenizerArtifactManager::last_error`] now keeps the whole message, so this classification is only exercised by the tests that pin the message↔code contract.
#[cfg(test)]
fn error_code(error: &Error) -> &'static str {
    let message = error.message();
    let head = message.split(':').next().unwrap_or_default().trim();
    for code in ARTIFACT_ERROR_CODES {
        if head == *code {
            return code;
        }
    }
    TOKENIZER_DOWNLOAD_FAILED_CODE
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(source: &str, id: &str, engine: TokenizerEngine) -> TokenizerSpec {
        TokenizerSpec {
            tokenizer_id: Some(id.to_owned()),
            source: Some(source.to_owned()),
            engine,
            ..TokenizerSpec::default()
        }
    }

    #[test]
    fn construction_is_network_free() {
        let manager = TokenizerArtifactManager::new();
        assert!(!manager.enable_download());
        assert!(!manager.offline());
        assert_eq!(manager.request_timeout(), DEFAULT_TOKENIZER_REQUEST_TIMEOUT);
        assert_eq!(manager.last_error(), None);
    }

    #[test]
    fn a_zero_timeout_restores_the_default() {
        let manager = TokenizerArtifactManager::new().with_request_timeout(Duration::ZERO);
        assert_eq!(manager.request_timeout(), DEFAULT_TOKENIZER_REQUEST_TIMEOUT);
    }

    #[test]
    fn cache_targets_depend_on_the_source_and_revision() {
        let manager = TokenizerArtifactManager::new().with_cache_dir("/cache");
        let hugging = manager
            .cache_target(&spec(
                HUGGINGFACE_SOURCE,
                "org/tok",
                TokenizerEngine::Native,
            ))
            .expect("target");
        assert_eq!(
            hugging,
            PathBuf::from("/cache/huggingface/org/tok/main/tokenizer.json")
        );
        let scoped = manager
            .cache_target(&spec(MODELSCOPE_SOURCE, "tok", TokenizerEngine::Tiktoken))
            .expect("target");
        assert_eq!(
            scoped,
            PathBuf::from("/cache/models/tok/master/tiktoken.model")
        );
        assert_eq!(
            manager.cache_target(&spec("local", "tok", TokenizerEngine::Auto)),
            None
        );
    }

    #[test]
    fn endpoint_builders_are_trimmed() {
        let manager = TokenizerArtifactManager::new().with_endpoint(" http://hub.local/ ");
        assert_eq!(manager.endpoint(), "http://hub.local");
        assert_eq!(manager.modelscope_endpoint(), "http://hub.local");
        let restored = manager.with_endpoint("");
        assert_eq!(restored.endpoint(), DEFAULT_HUGGINGFACE_ENDPOINT);
        assert_eq!(restored.modelscope_endpoint(), DEFAULT_MODELSCOPE_ENDPOINT);
    }

    #[test]
    fn error_codes_round_trip_through_the_message() {
        let error = Error::Validation(format!("{TOKENIZER_SHA256_MISMATCH_CODE}: detail"));
        assert_eq!(error_code(&error), TOKENIZER_SHA256_MISMATCH_CODE);
        assert_eq!(
            error_code(&Error::Validation("other".to_owned())),
            TOKENIZER_DOWNLOAD_FAILED_CODE
        );
    }

    #[test]
    fn a_url_id_is_refused_before_any_source() {
        let manager = TokenizerArtifactManager::new().with_cache_dir("/cache");
        let cached = manager.resolve_cached(&spec(
            HUGGINGFACE_SOURCE,
            "https://evil.example/tokenizer.json",
            TokenizerEngine::Native,
        ));
        assert_eq!(cached, None);
        assert_eq!(
            manager.last_error().as_deref(),
            Some(REMOTE_URL_NOT_ALLOWED_CODE)
        );
    }

    #[test]
    fn provider_official_and_local_specs_record_their_codes() {
        let manager = TokenizerArtifactManager::new().with_cache_dir("/cache");
        assert_eq!(
            manager.resolve_cached(&spec(
                PROVIDER_OFFICIAL_SOURCE,
                "tok",
                TokenizerEngine::Native
            )),
            None
        );
        let recorded = manager.last_error().expect("adapter message");
        assert!(
            recorded.starts_with(PROVIDER_OFFICIAL_ADAPTER_NOT_CONFIGURED_CODE),
            "the message keeps its compact code: {recorded}"
        );
        assert_eq!(
            manager.resolve_cached(&spec("local", "tok", TokenizerEngine::Native)),
            None
        );
        assert_eq!(
            manager.last_error().as_deref(),
            Some(LOCAL_ARTIFACT_MISSING_CODE)
        );
    }

    #[test]
    fn a_provider_official_message_names_the_vendor_and_the_id() {
        let mut named = spec(
            PROVIDER_OFFICIAL_SOURCE,
            "vendor-mini",
            TokenizerEngine::Native,
        );
        named.provider = "acme".to_owned();
        let message = provider_official_error(&named).message().into_owned();
        assert!(
            message.starts_with(PROVIDER_OFFICIAL_ADAPTER_NOT_CONFIGURED_CODE),
            "{message}"
        );
        assert!(message.contains("acme:vendor-mini"), "{message}");
        assert!(
            message.contains("needs a acme tokenizer adapter"),
            "{message}"
        );

        // An unrecognised provider is reported verbatim instead of guessed.
        named.provider = "not-a-known-vendor".to_owned();
        let message = provider_official_error(&named).message().into_owned();
        assert!(
            message.contains("not-a-known-vendor:vendor-mini"),
            "{message}"
        );

        // With no provider the id's `vendor/name` prefix names the vendor, and a bare id falls back to the neutral name.
        let from_id = provider_official_error(&spec(
            PROVIDER_OFFICIAL_SOURCE,
            "vendor/name",
            TokenizerEngine::Native,
        ))
        .message()
        .into_owned();
        assert!(from_id.contains("vendor:vendor/name"), "{from_id}");
        let unnamed = provider_official_error(&spec(
            PROVIDER_OFFICIAL_SOURCE,
            "unnamed",
            TokenizerEngine::Native,
        ))
        .message()
        .into_owned();
        assert!(unnamed.contains("provider-official:unnamed"), "{unnamed}");
    }
}
