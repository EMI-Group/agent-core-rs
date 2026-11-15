//! `OpenAI` account OAuth transport: device-code login, token refresh, and the
//! credential store.
//!
//! The module ports the `OpenAIAccount` provider of the reference SDK: the OAuth
//! 2.0 device authorization grant against `https://auth.openai.com`, the token
//! refresh that rotates the refresh token, and the private auth-store file that
//! keeps the credentials between runs.
//!
//! - [`OAuthToken`] — the token set, with a redacting `Debug` and a tolerant
//!   `serde` shape (`expires_in` becomes an absolute `expires_at`).
//! - [`TokenStore`], [`InMemoryTokenStore`], [`FileTokenStore`] — where the token
//!   is kept; the file backend merges one provider entry into the shared auth
//!   store instead of overwriting it.
//! - [`TokenFileLock`] — the exclusive cross-process lock (`flock` on
//!   `<auth_file>.lock`) that serializes the auth store's read-modify-write
//!   cycles, so two processes cannot clobber each other's entries.
//! - [`DeviceCode`] — the pending device authorization handed to the user.
//! - [`OpenAiOAuthClient`] — the HTTP side: request, poll, exchange, refresh, and
//!   the `access_token` shortcut that keeps the store in sync.
//! - [`login`], [`status`], [`logout`] — the whole-flow entry points a CLI needs,
//!   each taking one [`OAuthCredentials`] that pairs a store with the transport.
//!   [`login`] publishes the verification URL and user code through `tracing` and
//!   returns them as a [`LoginInfo`], [`status`] reports a [`TokenStatus`], and
//!   [`logout`] reports whether a credential was removed.
//!
//! # Expiry rule
//!
//! The expiry of a token response is resolved with the reference implementation's
//! precedence (`expires_at` → `expires_in` → the previous token's expiry → the
//! `exp` claim of a JWT access token). A token whose expiry resolves from **none**
//! of them is treated as already expired by [`OAuthToken::is_expired`], so
//! [`OpenAiOAuthClient::access_token`] attempts the refresh-token flow on the next
//! use instead of handing out a credential of unknown age forever; with no refresh
//! token stored, that attempt reports the re-login error.
//!
//! Only the token endpoints live here. Sending an `OpenAI` request with the
//! resulting bearer token — the Responses-API transport — is outside this module.
//!
//! ```no_run
//! use openjiuwen_foundation::llm::oauth::{self, OAuthCredentials};
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let credentials = OAuthCredentials::new()?;
//! if !oauth::status(&credentials).await?.is_valid() {
//!     // The verification URL and user code are also published through `tracing`.
//!     let info = oauth::login(&credentials).await?;
//!     let _ = (info.verification_uri, info.user_code);
//! }
//! let token = credentials
//!     .client()
//!     .access_token(credentials.store(), false)
//!     .await?;
//! # let _ = token;
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod credentials;
pub mod lock;
pub mod store;

#[cfg(test)]
mod credential_tests;
#[cfg(test)]
mod lock_tests;
#[cfg(test)]
mod refresh_tests;
#[cfg(test)]
mod support;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod token_tests;

pub use client::{
    OpenAiOAuthClient, DEFAULT_CLIENT_ID, DEFAULT_ISSUER, DEVICE_CALLBACK_PATH,
    DEVICE_VERIFICATION_PATH,
};
pub use credentials::{
    login, logout, status, LoginInfo, OAuthCredentials, TokenStatus, TokenSummary,
};
pub use lock::{lock_path, TokenFileLock, LOCK_TIMEOUT};
pub use store::{
    DeviceCode, FileTokenStore, InMemoryTokenStore, OAuthToken, TokenStore, AUTH_FILE_ENV,
    AUTH_MODE, DEFAULT_BASE_URL, PROVIDER_KEY, REFRESH_SKEW,
};
