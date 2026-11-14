//! `OpenAIAccount` credential manager over the shared OAuth token store.
//!
//! Ports the state half of
//! `openjiuwen/extensions/external_provider/openai_auth/openai_account_auth.py`:
//! [`OpenAiAccountAuthManager::status`], [`logout`](OpenAiAccountAuthManager::logout),
//! [`load_tokens`](OpenAiAccountAuthManager::load_tokens),
//! [`save_tokens`](OpenAiAccountAuthManager::save_tokens),
//! [`refresh_tokens`](OpenAiAccountAuthManager::refresh_tokens), and
//! [`resolve_access_token`](OpenAiAccountAuthManager::resolve_access_token).
//!
//! The device-authorization and refresh HTTP calls are the ones in
//! [`crate::llm::oauth`]; this module only decides *when* to call them and keeps
//! the store in sync. The refresh decision is the reference implementation's:
//! refresh when the access token expires within the configured skew (120 s by
//! default), or unconditionally when `force_refresh` is set.
//!
//! Two deliberate differences, both documented in the parent module's
//! `CONTEXT.md`:
//!
//! - there is no cross-process `FileLock`; [`FileTokenStore`] writes atomically
//!   (`tmp` + `rename`, `0600`) and the last writer wins, exactly as documented
//!   for [`crate::llm::oauth`];
//! - the stored `auth_mode` marker stays `openai_account_oauth` (the value
//!   [`crate::llm::oauth`] writes), not Python's `openai_account`; the token
//!   payload itself is byte-compatible.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::config::env;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_common::security::SecretString;

use crate::llm::oauth::{
    FileTokenStore, OAuthToken, OpenAiOAuthClient, TokenStore, REFRESH_SKEW,
};

use super::super::base::{AuthStatus, ExternalAuthProvider, ExternalProviderConfig};
use super::{BASE_URL_ENV, DEFAULT_BASE_URL};

/// Status code reported when the auth store holds no `OpenAIAccount` entry
/// (`openai_account_auth_missing`).
pub const MISSING_AUTH_CODE: &str = "openai_account_auth_missing";

/// Status code reported when the auth store cannot be parsed
/// (`openai_account_auth_invalid_store`).
pub const INVALID_STORE_CODE: &str = "openai_account_auth_invalid_store";

/// Status code reported when the auth store cannot be read
/// (`openai_account_auth_store_read_failed`).
pub const STORE_READ_FAILED_CODE: &str = "openai_account_auth_store_read_failed";

/// Manages the `OpenAIAccount` credential in openJiuwen's auth store.
///
/// A manager is cheap: it holds a [`TokenStore`] handle, an
/// [`OpenAiOAuthClient`], and the refresh skew. Every operation reads the store,
/// so several managers — or several processes — never disagree about the current
/// token.
///
/// The constructor resolves, in order:
///
/// 1. the auth-store path: [`ExternalProviderConfig::auth_path`], then
///    [`FileTokenStore::default_path`] (`$OPENJIUWEN_AUTH_FILE` →
///    `$OPENJIUWEN_HOME/auth.json` → `~/.openjiuwen/auth.json`);
/// 2. the account base URL: [`ExternalProviderConfig::base_url`], then
///    `$OPENJIUWEN_OPENAI_ACCOUNT_BASE_URL`, then [`DEFAULT_BASE_URL`];
/// 3. the refresh skew: [`ExternalProviderConfig::refresh_skew`], then
///    [`REFRESH_SKEW`] (120 s).
pub struct OpenAiAccountAuthManager {
    /// Where the token lives.
    store: Arc<dyn TokenStore>,
    /// The OAuth transport used for refreshes.
    client: OpenAiOAuthClient,
    /// Auth-store path reported by [`AuthStatus::auth_path`], when file-backed.
    auth_path: Option<PathBuf>,
    /// Account API base URL recorded next to the tokens.
    base_url: String,
    /// Refresh skew applied to the expiry check.
    refresh_skew: Duration,
}

impl OpenAiAccountAuthManager {
    /// Builds a manager from the environment and built-in defaults.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when no auth-store path can be resolved (no
    /// `OPENJIUWEN_AUTH_FILE`, no `OPENJIUWEN_HOME`, and no home directory) and
    /// the errors of [`OpenAiOAuthClient::new`].
    pub fn new() -> Result<Self> {
        Self::with_config(&ExternalProviderConfig::default())
    }

    /// Builds a manager from a typed provider configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when no auth-store path can be resolved and the
    /// errors of [`OpenAiOAuthClient::new`].
    pub fn with_config(config: &ExternalProviderConfig) -> Result<Self> {
        let auth_path = match &config.auth_path {
            Some(path) => expand_tilde(path),
            None => FileTokenStore::default_path()?,
        };
        let base_url = resolve_base_url(config.base_url.as_deref());
        let refresh_skew = config.refresh_skew_or(REFRESH_SKEW);
        let store = FileTokenStore::new(auth_path.clone()).with_base_url(base_url.clone());
        Ok(Self {
            store: Arc::new(store),
            client: OpenAiOAuthClient::new()?,
            auth_path: Some(auth_path),
            base_url,
            refresh_skew,
        })
    }

    /// Builds a manager over an arbitrary token store.
    ///
    /// The auth store has no path, so [`AuthStatus::auth_path`] stays `None`.
    /// Use [`with_oauth_client`](Self::with_oauth_client) to point refreshes at a
    /// test server.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`OpenAiOAuthClient::new`].
    pub fn with_store(store: Arc<dyn TokenStore>) -> Result<Self> {
        Ok(Self {
            store,
            client: OpenAiOAuthClient::new()?,
            auth_path: None,
            base_url: resolve_base_url(None),
            refresh_skew: REFRESH_SKEW,
        })
    }

    /// Replaces the OAuth client (issuer, client id, HTTP stack).
    #[must_use]
    pub fn with_oauth_client(mut self, client: OpenAiOAuthClient) -> Self {
        self.client = client;
        self
    }

    /// Overrides the account API base URL recorded next to the tokens.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        base_url.trim().trim_end_matches('/').clone_into(&mut self.base_url);
        self
    }

    /// Overrides the refresh skew.
    #[must_use]
    pub fn with_refresh_skew(mut self, skew: Duration) -> Self {
        self.refresh_skew = skew;
        self
    }

    /// Returns the auth-store path, or `None` for a custom token store.
    #[must_use]
    pub fn auth_path(&self) -> Option<&Path> {
        self.auth_path.as_deref()
    }

    /// Returns the account API base URL recorded next to the tokens.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns the refresh skew.
    #[must_use]
    pub fn refresh_skew(&self) -> Duration {
        self.refresh_skew
    }

    /// Returns the current credential status.
    ///
    /// A missing credential, an unreadable store, and a corrupt store all become
    /// `authenticated: false` with the matching [`AuthStatus::error`] code; the
    /// call itself does not fail for them (Python parity).
    ///
    /// # Errors
    ///
    /// Only for failures a status cannot express; the built-in file store always
    /// reports I/O and parse problems through the status instead.
    pub async fn status(&self) -> Result<AuthStatus> {
        match self.store.load().await {
            Ok(Some(token)) => Ok(AuthStatus {
                authenticated: true,
                auth_path: self.auth_path.clone(),
                has_refresh_token: token
                    .refresh_token
                    .as_ref()
                    .is_some_and(|value| !value.expose_secret().trim().is_empty()),
                expires_at: token.effective_expires_at(),
                needs_refresh: token.is_expired(self.refresh_skew),
                error: None,
            }),
            Ok(None) => Ok(AuthStatus::unauthenticated(
                self.auth_path.clone(),
                MISSING_AUTH_CODE,
            )),
            Err(error) => {
                let code = store_error_code(&error);
                tracing::debug!(
                    target: "openjiuwen_foundation::llm::external_provider",
                    error = %error,
                    code,
                    "reading the OpenAI account credential store failed"
                );
                Ok(AuthStatus::unauthenticated(self.auth_path.clone(), code))
            }
        }
    }

    /// Removes the stored credential, returning `true` when one was present.
    ///
    /// # Errors
    ///
    /// Returns the store error when the credential store cannot be read or
    /// written.
    pub async fn logout(&self) -> Result<bool> {
        if self.store.load().await?.is_none() {
            return Ok(false);
        }
        self.store.clear().await?;
        Ok(true)
    }

    /// Loads the stored token.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when no credential is stored (the Python code is
    /// [`MISSING_AUTH_CODE`]; call [`status`](Self::status) to read it) and the
    /// store error when the store is unreadable or corrupt.
    pub async fn load_tokens(&self) -> Result<OAuthToken> {
        self.store.load().await?.ok_or_else(|| {
            Error::config(
                "No OpenAI account credentials stored; run the device authorization flow first.",
            )
        })
    }

    /// Stores `token`, replacing any previous credential.
    ///
    /// # Errors
    ///
    /// Returns the store error when the credential store cannot be written.
    pub async fn save_tokens(&self, token: &OAuthToken) -> Result<()> {
        self.store.save(token).await
    }

    /// Returns the current token, refreshing and persisting it when needed.
    ///
    /// A token that is still valid is returned unchanged unless `force` is set,
    /// in which case the refresh endpoint is called even then (Python parity).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when no credential is stored, the errors of
    /// [`OpenAiOAuthClient::refresh`] when the refresh is rejected, and the store
    /// error when the refreshed token cannot be persisted.
    pub async fn refresh_tokens(&self, force: bool) -> Result<OAuthToken> {
        let current = self.load_tokens().await?;
        if !force && !current.is_expired(self.refresh_skew) {
            return Ok(current);
        }
        let refreshed = self.client.refresh(&current).await?;
        self.store.save(&refreshed).await?;
        Ok(refreshed)
    }

    /// Returns a usable access token, refreshing it when needed.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`refresh_tokens`](Self::refresh_tokens).
    pub async fn resolve_access_token(&self, force_refresh: bool) -> Result<SecretString> {
        Ok(self.refresh_tokens(force_refresh).await?.access_token)
    }
}

#[async_trait]
impl ExternalAuthProvider for OpenAiAccountAuthManager {
    async fn status(&self) -> Result<AuthStatus> {
        OpenAiAccountAuthManager::status(self).await
    }

    async fn logout(&self) -> Result<bool> {
        OpenAiAccountAuthManager::logout(self).await
    }

    async fn resolve_access_token(&self, force_refresh: bool) -> Result<SecretString> {
        OpenAiAccountAuthManager::resolve_access_token(self, force_refresh).await
    }
}

impl fmt::Debug for OpenAiAccountAuthManager {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // No token is reachable from the manager itself, so listing the
        // configuration is safe; the store is described by its path only.
        formatter
            .debug_struct("OpenAiAccountAuthManager")
            .field("auth_path", &self.auth_path)
            .field("base_url", &self.base_url)
            .field("refresh_skew", &self.refresh_skew)
            .finish_non_exhaustive()
    }
}

/// Maps a store error to the Python status code.
///
/// [`ErrorKind::External`] means reading the file failed; every other kind means
/// the store was readable but unusable (invalid JSON, a non-object document, or a
/// malformed token entry), which is the reference implementation's
/// `openai_account_auth_invalid_store` case.
fn store_error_code(error: &Error) -> &'static str {
    match error.kind() {
        ErrorKind::External => STORE_READ_FAILED_CODE,
        _ => INVALID_STORE_CODE,
    }
}

/// Resolves the account base URL: config, then `$OPENJIUWEN_OPENAI_ACCOUNT_BASE_URL`,
/// then [`DEFAULT_BASE_URL`].
fn resolve_base_url(configured: Option<&str>) -> String {
    configured
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            env::var(BASE_URL_ENV)
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned())
        .trim_end_matches('/')
        .to_owned()
}

/// Expands a leading `~` in `path` using `$HOME`/`%USERPROFILE%`.
///
/// Python's `Path.expanduser()`; a path without a leading `~` is returned
/// unchanged, and an unresolvable `~` also stays literal rather than failing the
/// construction (the store then reports a read error through
/// [`OpenAiAccountAuthManager::status`]).
fn expand_tilde(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if raw == "~" || raw.starts_with("~/") {
        if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
            let home = PathBuf::from(home);
            if !home.as_os_str().is_empty() {
                return home.join(raw.trim_start_matches("~/"));
            }
        }
    }
    path.to_path_buf()
}
