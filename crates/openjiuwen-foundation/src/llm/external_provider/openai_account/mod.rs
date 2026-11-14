//! Built-in `OpenAIAccount` external provider: credential manager and model catalog.
//!
//! Ports `openjiuwen/extensions/external_provider/openai_auth/` onto the OAuth
//! machinery that already lives in [`crate::llm::oauth`]:
//!
//! - [`OpenAiAccountAuthManager`] implements [`ExternalAuthProvider`] over a
//!   [`TokenStore`](crate::llm::oauth::TokenStore) (the shared auth file by
//!   default) and [`OpenAiOAuthClient`](crate::llm::oauth::OpenAiOAuthClient)
//!   (device login, token exchange, refresh).
//! - [`OpenAiAccountModelCatalog`] implements [`ProviderModelCatalog`] with the
//!   live → cache → built-in-defaults precedence of the reference
//!   implementation.
//!
//! Only the OAuth *state* surface lives here: the device-code login itself is
//! already part of [`crate::llm::oauth`], so the manager exposes
//! [`OpenAiAccountAuthManager::refresh_tokens`] and delegates credential
//! resolution to it instead of duplicating the HTTP flow.
//!
//! The constants below keep the reference implementation's names and values, so
//! an auth store or cache file written by the Python SDK is readable here (the
//! `auth_mode` marker inside the store entry differs; see the module's
//! `CONTEXT.md`).
//!
//! [`ExternalAuthProvider`]: super::base::ExternalAuthProvider
//! [`ProviderModelCatalog`]: super::base::ProviderModelCatalog

pub mod auth;
pub mod catalog;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use openjiuwen_common::error::Result;

use super::base::{ExternalAuthProvider, ExternalProviderConfig, ProviderModelCatalog};

pub use auth::{
    OpenAiAccountAuthManager, INVALID_STORE_CODE, MISSING_AUTH_CODE, STORE_READ_FAILED_CODE,
};
pub use catalog::{
    default_cache_path, forward_compat_models, parse_model_ids, OpenAiAccountModelCatalog,
    DEFAULT_TIMEOUT_SECONDS,
};

/// Provider name the built-ins register under (`OPENAI_ACCOUNT_PROVIDER`).
pub const PROVIDER_NAME: &str = "OpenAIAccount";

/// Account API base URL used when neither config nor environment overrides it
/// (`DEFAULT_OPENAI_ACCOUNT_BASE_URL`).
pub const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";

/// Access tokens are refreshed once they expire within this many seconds
/// (`OPENAI_ACCOUNT_ACCESS_TOKEN_REFRESH_SKEW_SECONDS`).
pub const REFRESH_SKEW_SECONDS: u64 = 120;

/// `client_version` query value sent to the model endpoint
/// (`OPENAI_ACCOUNT_MODELS_CLIENT_VERSION`).
pub const MODELS_CLIENT_VERSION: &str = "1.0.0";

/// File name of the model-list cache inside the auth-store directory
/// (`OPENAI_ACCOUNT_MODELS_CACHE_FILENAME`).
pub const MODELS_CACHE_FILENAME: &str = "openai_account_models_cache.json";

/// Environment variable overriding the model-cache location
/// (`OPENJIUWEN_OPENAI_ACCOUNT_MODELS_CACHE_FILE`).
pub const MODELS_CACHE_FILE_ENV: &str = "OPENJIUWEN_OPENAI_ACCOUNT_MODELS_CACHE_FILE";

/// Environment variable overriding the account API base URL
/// (`OPENJIUWEN_OPENAI_ACCOUNT_BASE_URL`).
pub const BASE_URL_ENV: &str = "OPENJIUWEN_OPENAI_ACCOUNT_BASE_URL";

/// Built-in model IDs used when neither the live endpoint nor a cache answers
/// (`DEFAULT_OPENAI_ACCOUNT_MODELS`), in the reference implementation's order.
pub const DEFAULT_MODELS: [&str; 6] = [
    "gpt-5.5",
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5-codex",
];

/// Builds an [`OpenAiAccountAuthManager`] from a typed configuration.
///
/// Used as the built-in auth provider factory registered by
/// [`ExternalProviderRegistry::with_builtins`](super::registry::ExternalProviderRegistry::with_builtins).
///
/// # Errors
///
/// Returns the errors of [`OpenAiAccountAuthManager::with_config`], for example
/// a missing home directory when no auth path can be resolved.
pub fn auth_provider_factory(
    config: &ExternalProviderConfig,
) -> Result<Arc<dyn ExternalAuthProvider>> {
    Ok(Arc::new(OpenAiAccountAuthManager::with_config(config)?))
}

/// Builds an [`OpenAiAccountModelCatalog`] from a typed configuration.
///
/// # Errors
///
/// Returns the errors of [`OpenAiAccountModelCatalog::with_config`], for example
/// a missing home directory when no cache path can be resolved.
pub fn model_catalog_factory(
    config: &ExternalProviderConfig,
) -> Result<Arc<dyn ProviderModelCatalog>> {
    Ok(Arc::new(OpenAiAccountModelCatalog::with_config(config)?))
}
