//! Device-code login, credential status, and logout for the `OpenAI` account.
//!
//! These are the three entry points a CLI (or any host process) needs, and each
//! takes one [`OAuthCredentials`] describing *where* the credential lives and
//! *which* OAuth transport to talk to:
//!
//! - [`login`] runs the whole device authorization grant — request a user code,
//!   publish the verification URL and the code through `tracing` (and return them
//!   in the [`LoginInfo`] so the caller can print them), poll until the user signs
//!   in, exchange the authorization code, and persist the token under the
//!   cross-process lock;
//! - [`status`] reports the stored credential as a [`TokenStatus`] (path,
//!   present/absent, expiry, expired/valid, refresh-token presence, auth mode);
//! - [`logout`] removes the persisted credential and reports whether one was
//!   there, so repeated logouts are `Ok(false)` rather than errors.
//!
//! The three deliberately sit above [`OpenAiOAuthClient`] and the [`TokenStore`]
//! seam: no HTTP call is duplicated here, and both the file-backed store (the
//! production case, serialized through
//! [`TokenFileLock`](super::lock::TokenFileLock)) and any custom store (memory,
//! keyring, secret manager) work unchanged. [`TokenStatus`] is the
//! `oauth`-owned status snapshot; the provider-facing
//! [`AuthStatus`](crate::llm::external_provider::AuthStatus) stays where it is and
//! is not duplicated here.
//!
//! No credential is ever logged: the user code and the verification URL are public
//! by design, while the access, refresh, and identity tokens are
//! [`SecretString`](openjiuwen_common::security::SecretString) values that never
//! leave [`TokenSummary`] as anything but presence flags and metadata.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};

use crate::llm::transport::join_url;

use super::client::{OpenAiOAuthClient, DEVICE_VERIFICATION_PATH};
use super::store::{auth_mode_async, FileTokenStore, OAuthToken, TokenStore, REFRESH_SKEW};

/// Logging target of this module's status and login events.
const LOG_TARGET: &str = "openjiuwen_foundation::llm::oauth";

/// The credential target and OAuth transport behind [`login`], [`status`], and
/// [`logout`].
///
/// A file-backed credential (the shipped `auth.json`) is serialized through the
/// cross-process [`TokenFileLock`](super::lock::TokenFileLock); any other
/// [`TokenStore`] is used as given and has no path to report or lock.
///
/// The type is cheap to clone in spirit (it holds a store handle and a client);
/// building one resolves the HTTP stack, hence the fallible constructors.
pub struct OAuthCredentials {
    /// Where the token is read from and written to.
    target: CredentialTarget,
    /// The OAuth transport (issuer, client id, HTTP stack).
    client: OpenAiOAuthClient,
    /// The skew the status/refresh decision uses.
    skew: Duration,
}

/// The store a credential operation targets.
enum CredentialTarget {
    /// The auth-store file, whose writes take the cross-process lock.
    File(FileTokenStore),
    /// Any other store (memory, keyring, secret manager): no file, no lock.
    Store(Arc<dyn TokenStore>),
}

impl OAuthCredentials {
    /// Targets the default auth-store path and the built-in issuer.
    ///
    /// The path is [`FileTokenStore::default_path`]: `$OPENJIUWEN_AUTH_FILE`, then
    /// `$OPENJIUWEN_HOME/auth.json`, then `~/.openjiuwen/auth.json`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](openjiuwen_common::error::Error::Config) when no
    /// auth-store path resolves, and the errors of
    /// [`OpenAiOAuthClient::new`].
    pub fn new() -> Result<Self> {
        Self::at_path(FileTokenStore::default_path()?)
    }

    /// Targets the auth-store file at `path` with the built-in issuer.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`OpenAiOAuthClient::new`].
    pub fn at_path(path: impl Into<PathBuf>) -> Result<Self> {
        Self::from_file(FileTokenStore::new(path))
    }

    /// Targets an explicit file-backed store (path and recorded `base_url`).
    ///
    /// # Errors
    ///
    /// Returns the errors of [`OpenAiOAuthClient::new`].
    pub fn from_file(store: FileTokenStore) -> Result<Self> {
        Ok(Self {
            target: CredentialTarget::File(store),
            client: OpenAiOAuthClient::new()?,
            skew: REFRESH_SKEW,
        })
    }

    /// Targets a custom token store (memory, keyring, secret manager).
    ///
    /// # Errors
    ///
    /// Returns the errors of [`OpenAiOAuthClient::new`].
    pub fn from_store(store: Arc<dyn TokenStore>) -> Result<Self> {
        Ok(Self {
            target: CredentialTarget::Store(store),
            client: OpenAiOAuthClient::new()?,
            skew: REFRESH_SKEW,
        })
    }

    /// Replaces the OAuth transport (issuer, client id, HTTP stack).
    #[must_use]
    pub fn with_client(mut self, client: OpenAiOAuthClient) -> Self {
        self.client = client;
        self
    }

    /// Replaces the refresh skew the [`status`] decision uses.
    #[must_use]
    pub fn with_skew(mut self, skew: Duration) -> Self {
        self.skew = skew;
        self
    }

    /// Returns the token store the operations drive.
    #[must_use]
    pub fn store(&self) -> &dyn TokenStore {
        match &self.target {
            CredentialTarget::File(store) => store,
            CredentialTarget::Store(store) => store.as_ref(),
        }
    }

    /// Returns the auth-store path, or `None` for a custom token store.
    #[must_use]
    pub fn auth_path(&self) -> Option<&Path> {
        match &self.target {
            CredentialTarget::File(store) => Some(store.path()),
            CredentialTarget::Store(_) => None,
        }
    }

    /// Returns the OAuth transport.
    #[must_use]
    pub fn client(&self) -> &OpenAiOAuthClient {
        &self.client
    }

    /// Returns the refresh skew used by [`status`].
    #[must_use]
    pub fn skew(&self) -> Duration {
        self.skew
    }
}

impl fmt::Debug for OAuthCredentials {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The stores hold no `Debug`-able token (the client prints only its
        // issuer), so listing the configuration is safe.
        formatter
            .debug_struct("OAuthCredentials")
            .field("auth_path", &self.auth_path())
            .field("client", &self.client)
            .field("skew", &self.skew)
            .finish_non_exhaustive()
    }
}

/// A completed device-code login.
///
/// The verification URL and user code are returned so the caller can show them
/// even when it does not read the log; [`token`](Self::token) is a summary, never
/// the credential itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoginInfo {
    /// URL the user opens to authorize the login.
    pub verification_uri: String,
    /// Code the user types into that page.
    pub user_code: String,
    /// How long the code stays valid, in seconds (`0` means "not advertised").
    pub expires_in: u64,
    /// The polling interval the issuer advertised, in seconds.
    pub interval: u64,
    /// Summary of the token that was persisted.
    pub token: TokenSummary,
}

/// The non-secret shape of a stored token.
///
/// The access, refresh, and identity tokens are reported only as presence flags (or
/// not at all), so a summary can be logged or serialized without leaking a
/// credential.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenSummary {
    /// Expiry the refresh decision uses, when one resolves.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "chrono::serde::ts_seconds_option"
    )]
    pub expires_at: Option<DateTime<Utc>>,
    /// The token type reported by the issuer (`Bearer` by default).
    pub token_type: String,
    /// The granted OAuth scopes, when the issuer reported them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Whether a refresh token is stored alongside the access token.
    pub has_refresh_token: bool,
    /// Whether an `OpenID` Connect identity token is stored.
    pub has_id_token: bool,
}

impl TokenSummary {
    /// Summarizes `token` without copying any credential.
    #[must_use]
    pub fn of(token: &OAuthToken) -> Self {
        Self {
            expires_at: token.effective_expires_at(),
            token_type: token.token_type.clone(),
            scope: token.scope.clone(),
            has_refresh_token: has_refresh_token(token),
            has_id_token: token.id_token.is_some(),
        }
    }
}

/// The state of the credential behind an [`OAuthCredentials`] target.
///
/// This is the `oauth`-owned snapshot; the provider-facing
/// [`AuthStatus`](crate::llm::external_provider::AuthStatus) keeps its own shape
/// and error codes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenStatus {
    /// Credential file the status was read from, `None` for a store with no file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_path: Option<PathBuf>,
    /// Whether a credential is stored.
    pub present: bool,
    /// Expiry the refresh decision uses, when one resolves.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "chrono::serde::ts_seconds_option"
    )]
    pub expires_at: Option<DateTime<Utc>>,
    /// Whether the credential is stale at the configured skew.
    ///
    /// A credential whose expiry cannot be resolved is stale (see
    /// [`OAuthToken::is_expired`]), so it reports `true` here while
    /// [`expires_at`](Self::expires_at) stays `None`.
    pub expired: bool,
    /// Whether a refresh token is stored alongside the access token.
    pub has_refresh_token: bool,
    /// The `auth_mode` marker recorded in the store entry, when there is one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_mode: Option<String>,
}

impl TokenStatus {
    /// Returns whether a stored credential could be used as-is right now.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.present && !self.expired
    }
}

/// Runs the device-authorization login and persists the resulting token.
///
/// The flow is the standard OAuth 2.0 device grant, and every step is the matching
/// [`OpenAiOAuthClient`] call:
///
/// 1. [`request_device_code`](OpenAiOAuthClient::request_device_code) asks the
///    issuer for a user code;
/// 2. the verification URL (`{issuer}/codex/device`) and that code are published
///    through `tracing` and returned in the [`LoginInfo`], which is all a CLI needs
///    to show the user;
/// 3. [`poll_device_token`](OpenAiOAuthClient::poll_device_token) waits for the
///    sign-in and exchanges the authorization code;
/// 4. the token is persisted through the credential target — under the
///    cross-process [`TokenFileLock`](super::lock::TokenFileLock) for the
///    auth-store file — before this function returns.
///
/// The poll is bounded by the code's advertised lifetime, so a user who never signs
/// in yields [`Error::Llm`](openjiuwen_common::error::Error::Llm) instead of a
/// hanging call.
///
/// # Errors
///
/// Returns [`Error::Llm`](openjiuwen_common::error::Error::Llm) when the issuer
/// rejects a request, the device code expires, or a response is not a token, and the
/// store error when the token cannot be persisted.
pub async fn login(credentials: &OAuthCredentials) -> Result<LoginInfo> {
    let device = credentials.client.request_device_code().await?;
    let verification_uri = join_url(credentials.client.base_url(), DEVICE_VERIFICATION_PATH);
    tracing::info!(
        target: LOG_TARGET,
        verification_uri = %verification_uri,
        user_code = %device.user_code,
        expires_in = device.expires_in,
        interval = device.interval,
        "OpenAI account device login pending: open the verification URL and enter the user code"
    );

    let token = credentials.client.poll_device_token(&device).await?;
    credentials.store().save(&token).await?;
    let summary = TokenSummary::of(&token);
    tracing::info!(
        target: LOG_TARGET,
        auth_path = ?credentials.auth_path(),
        expires_at = ?summary.expires_at,
        has_refresh_token = summary.has_refresh_token,
        "OpenAI account login completed"
    );

    Ok(LoginInfo {
        verification_uri,
        user_code: device.user_code,
        expires_in: device.expires_in,
        interval: device.interval,
        token: summary,
    })
}

/// Reports the stored credential, reading the auth-store file under its lock.
///
/// # Errors
///
/// Returns the store error when the credential cannot be read (an unreadable or
/// corrupt auth-store document), and the lock errors of
/// [`FileTokenStore::lock`] for the file-backed target.
pub async fn status(credentials: &OAuthCredentials) -> Result<TokenStatus> {
    match &credentials.target {
        CredentialTarget::File(store) => {
            let path = store.path().to_path_buf();
            // The mode marker lives in the same document, so read it under the
            // same lock as the token.
            if let Some(lock) = store.lock_optional().await? {
                let mode = auth_mode_async(&path).await?;
                Ok(snapshot(
                    Some(path),
                    lock.load().await?,
                    credentials.skew,
                    mode,
                ))
            } else {
                let mode = auth_mode_async(&path).await?;
                Ok(snapshot(
                    Some(path),
                    store.load().await?,
                    credentials.skew,
                    mode,
                ))
            }
        }
        CredentialTarget::Store(store) => {
            Ok(snapshot(None, store.load().await?, credentials.skew, None))
        }
    }
}

/// Removes the persisted credential, reporting whether one was present.
///
/// The removal is idempotent: an absent credential — no file, no entry, or no
/// token in a custom store — is `Ok(false)`, not an error.
///
/// # Errors
///
/// Returns the store error when the credential cannot be read or removed, and the
/// lock errors of [`FileTokenStore::lock`] for the file-backed target.
pub async fn logout(credentials: &OAuthCredentials) -> Result<bool> {
    match &credentials.target {
        CredentialTarget::File(store) => match store.lock_optional().await? {
            Some(lock) => lock.clear().await,
            None => clear_through(store).await,
        },
        CredentialTarget::Store(store) => clear_through(store.as_ref()).await,
    }
}

/// Builds the status snapshot for a token read from `auth_path`.
fn snapshot(
    auth_path: Option<PathBuf>,
    token: Option<OAuthToken>,
    skew: Duration,
    auth_mode: Option<String>,
) -> TokenStatus {
    match token {
        Some(token) => TokenStatus {
            auth_path,
            present: true,
            expires_at: token.effective_expires_at(),
            expired: token.is_expired(skew),
            has_refresh_token: has_refresh_token(&token),
            auth_mode,
        },
        None => TokenStatus {
            auth_path,
            present: false,
            expires_at: None,
            expired: false,
            has_refresh_token: false,
            auth_mode,
        },
    }
}

/// Clears `store` in two steps, for backends without a locked `clear`.
async fn clear_through(store: &dyn TokenStore) -> Result<bool> {
    if store.load().await?.is_none() {
        return Ok(false);
    }
    store.clear().await?;
    Ok(true)
}

/// Whether the token carries a non-blank refresh token.
fn has_refresh_token(token: &OAuthToken) -> bool {
    token
        .refresh_token
        .as_ref()
        .is_some_and(|value| !value.expose_secret().trim().is_empty())
}
