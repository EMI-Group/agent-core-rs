//! External provider registry and the `OpenAIAccount` credential/catalog surface.
//!
//! Ports `openjiuwen/extensions/external_provider/` — the Python extension that
//! plugs a provider whose credentials live outside the SDK (currently the
//! `OpenAIAccount` `ChatGPT` account provider) into two shared seams:
//!
//! - [`ExternalAuthProvider`] — status, logout, and access-token resolution;
//! - [`ProviderModelCatalog`] — model discovery with live → cache → defaults
//!   fallback.
//!
//! [`ExternalProviderRegistry`] registers factories for both seams by provider
//! name and creates instances from a typed [`ExternalProviderConfig`]; the
//! process-global registry is what the free functions
//! ([`create_auth_provider`], [`create_model_catalog`], …) drive.
//!
//! The built-in [`OpenAiAccountAuthManager`](openai_account::OpenAiAccountAuthManager)
//! and [`OpenAiAccountModelCatalog`](openai_account::OpenAiAccountModelCatalog)
//! are wired into [`ExternalProviderRegistry::with_builtins`], so lookups work
//! without an explicit registration.
//!
//! # Layout
//!
//! - [`base`] — traits, [`AuthStatus`], [`ExternalProviderConfig`], and
//!   [`ModelCatalogRequest`].
//! - [`registry`] — the registry, the process-global instance, and the free
//!   functions.
//! - [`openai_account`] — the built-in provider: constants, credential manager,
//!   model catalog, and the pure model-list parser.
//!
//! # Relationship to `llm::oauth`
//!
//! The device-authorization grant, the token refresh, and the auth-store file
//! already live in [`crate::llm::oauth`]. This module composes them; it does not
//! duplicate a single HTTP call. The two differences from the Python extension
//! are documented in the `openai_account` module: no cross-process `FileLock`
//! (atomic writes, last writer wins) and the auth-store `auth_mode` marker stays
//! `openai_account_oauth`.

pub mod base;
pub mod openai_account;
pub mod registry;

#[cfg(test)]
mod registry_tests;

pub use base::{
    AuthStatus, ExternalAuthProvider, ExternalProviderConfig, ModelCatalogRequest,
    ProviderModelCatalog,
};
pub use registry::{
    create_auth_provider, create_model_catalog, get_auth_provider_factory,
    get_model_catalog_factory, list_auth_providers, list_model_catalogs,
    register_external_auth_provider, register_external_model_catalog, AuthProviderFactory,
    ExternalProviderRegistry, ModelCatalogFactory,
};
