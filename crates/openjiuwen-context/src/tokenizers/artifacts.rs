//! Tokenizer artefact discovery, caching and download.
//!
//! Ports `token/tokenizer_manager.py`: cache-directory precedence, local and
//! downloaded artefacts, `huggingface`/`modelscope` sources, sha256 validation,
//! file locking and the compact error codes the selector reports.
//!
//! The manager is deliberately network-free by construction: both download
//! switches default to off, so creating one and calling
//! [`TokenizerArtifactManager::resolve_cached`] never touches the network.
//! Downloads additionally require [`TokenizerArtifactManager::with_enable_download`]
//! and the absence of [`TokenizerArtifactManager::with_offline`], and the
//! selector seam only ever uses the cached rung.
//!
//! # Cache layout
//!
//! ```text
//! <cache_dir>/<cache key>.lock                       advisory download lock
//! <cache_dir>/huggingface/<tokenizer id>/<revision>/<artefact>
//! <cache_dir>/models/<tokenizer id>/<revision>/<artefact>
//! ```
//!
//! `<artefact>` is `tiktoken.model` for [`TokenizerEngine::Tiktoken`](crate::tokenizers::TokenizerEngine::Tiktoken) and
//! `tokenizer.json` otherwise, and `<revision>` is the spec's revision or the
//! hub branch (`main` for `huggingface`, `master` for `modelscope`). The
//! `models` prefix of the `modelscope` layout is also one of the three offline
//! snapshot layouts the resolver probes, so a downloaded `modelscope` artefact
//! is found again without a second request.
//!
//! The cache directory itself is resolved as: explicit argument →
//! `OPENJIUWEN_TOKENIZER_CACHE_DIR` → `~/.cache/openjiuwen/tokenizers` →
//! the relative `.cache/openjiuwen/tokenizers` when `HOME` is unset (a
//! documented fallback rather than a temp directory, because a cache should
//! survive the process).
mod cache;
mod lock;
mod manager;

pub use cache::{
    artifact_name, cache_dir_from, default_cache_dir, expand_home, find_artifact,
    relative_artifact_id, sha256_file, sha256_hex, CACHE_DIR_ENV, DEFAULT_CACHE_SUBDIR,
    HF_ENDPOINT_ENV, MAX_ARTIFACT_SEARCH_DEPTH,
};
pub use lock::{ArtifactLock, LOCK_WAIT_LIMIT, STALE_LOCK_AFTER};
pub use manager::{
    TokenizerArtifactManager, ARTIFACT_ERROR_CODES, DEFAULT_HUGGINGFACE_ENDPOINT,
    DEFAULT_MODELSCOPE_ENDPOINT, DEFAULT_TOKENIZER_REQUEST_TIMEOUT, HUGGINGFACE_DEFAULT_REVISION,
    HUGGINGFACE_SOURCE, LOCAL_ARTIFACT_MISSING_CODE, MAX_ARTIFACT_BYTES,
    MODELSCOPE_DEFAULT_REVISION, MODELSCOPE_SOURCE, PROVIDER_OFFICIAL_ADAPTER_NOT_CONFIGURED_CODE,
    PROVIDER_OFFICIAL_SOURCE, REMOTE_DOWNLOAD_DISABLED_CODE, REMOTE_URL_NOT_ALLOWED_CODE,
    TIKTOKEN_MODEL_MISSING_CODE, TOKENIZER_DOWNLOAD_FAILED_CODE, TOKENIZER_ID_MISSING_CODE,
    TOKENIZER_JSON_MISSING_CODE, TOKENIZER_NETWORK_UNAVAILABLE_CODE,
    TOKENIZER_NOT_IN_LOCAL_CACHE_CODE, TOKENIZER_REVISION_UNAVAILABLE_CODE,
    TOKENIZER_SHA256_MISMATCH_CODE,
};
