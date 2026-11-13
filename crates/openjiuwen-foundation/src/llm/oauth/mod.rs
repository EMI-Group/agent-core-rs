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
//! - [`DeviceCode`] — the pending device authorization handed to the user.
//! - [`OpenAiOAuthClient`] — the HTTP side: request, poll, exchange, refresh, and
//!   the `access_token` shortcut that keeps the store in sync.
//!
//! Only the token endpoints live here. Sending an `OpenAI` request with the
//! resulting bearer token — the Responses-API transport — is outside this module.
//!
//! ```no_run
//! use openjiuwen_foundation::llm::oauth::{FileTokenStore, OpenAiOAuthClient, TokenStore};
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let client = OpenAiOAuthClient::new()?;
//! let store = FileTokenStore::new(FileTokenStore::default_path()?);
//! if store.load().await?.is_none() {
//!     let device_code = client.request_device_code().await?;
//!     // Show `device_code.user_code` to the user, then wait for the sign-in.
//!     let token = client.poll_device_token(&device_code).await?;
//!     store.save(&token).await?;
//! }
//! let token = client.access_token(&store, false).await?;
//! # let _ = token;
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod store;

#[cfg(test)]
mod refresh_tests;
#[cfg(test)]
mod support;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod token_tests;

pub use client::{OpenAiOAuthClient, DEFAULT_CLIENT_ID, DEFAULT_ISSUER, DEVICE_CALLBACK_PATH};
pub use store::{
    DeviceCode, FileTokenStore, InMemoryTokenStore, OAuthToken, TokenStore, AUTH_FILE_ENV,
    AUTH_MODE, DEFAULT_BASE_URL, PROVIDER_KEY, REFRESH_SKEW,
};
