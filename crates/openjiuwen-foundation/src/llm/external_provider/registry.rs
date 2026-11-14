//! Pluggable registry of external auth providers and model catalogs.
//!
//! Ports `openjiuwen/extensions/external_provider/registry.py`: provider names
//! are trimmed and case-folded, duplicate registration without `replace` is an
//! error naming the provider, an unknown lookup is an error listing the
//! available providers, and the built-in `OpenAIAccount` provider is registered
//! so lookups work without an explicit registration.
//!
//! Three Rust-specific choices:
//!
//! - Python's `ExternalProviderRegistryError(ValueError)` becomes
//!   [`openjiuwen_common::error::Error::Validation`], the crate's taxonomy for
//!   registry and configuration problems.
//! - Python's keyword-only `replace: bool = False` is a positional argument, and
//!   registration returns the factory it replaced (like
//!   [`ModelProviderRegistry::register`](crate::llm::ModelProviderRegistry::register))
//!   so a caller can restore it.
//! - Provider keys additionally ignore `-`, `_`, and whitespace, matching
//!   [`normalize_provider_name`](crate::llm::normalize_provider_name): the
//!   built-in provider therefore answers to `OpenAIAccount`, `openai_account`,
//!   `openai-account`, and `openaiaccount` alike. Python's key is only
//!   `strip().lower()`, so `openai_account` would not match there.
//!
//! # The process-global registry
//!
//! [`ExternalProviderRegistry::global`] is the second sanctioned mutable process
//! global in this crate, next to
//! [`ModelProviderRegistry::global`](crate::llm::ModelProviderRegistry::global).
//! It is created on first access with the built-in providers already registered
//! and is guarded by an `RwLock`, so registrations are visible to every caller
//! and reads take a short-lived lock. Tests and components that must not depend
//! on global state build an isolated instance with
//! [`ExternalProviderRegistry::new`] or [`ExternalProviderRegistry::with_builtins`].
//!
//! ```no_run
//! use openjiuwen_foundation::llm::external_provider::{
//!     ExternalAuthProvider, ExternalProviderConfig, ExternalProviderRegistry,
//! };
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let registry = ExternalProviderRegistry::with_builtins();
//! let auth = registry.create_auth_provider("openai_account", &ExternalProviderConfig::default())?;
//! assert!(auth.status().await?.error.is_some(), "no credential is stored in a fresh environment");
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, LazyLock};

use openjiuwen_common::error::{Error, Result};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use super::base::{ExternalAuthProvider, ExternalProviderConfig, ProviderModelCatalog};
use super::openai_account::{self, PROVIDER_NAME};

/// Builds an auth provider from a typed configuration.
///
/// Replaces Python's `Callable[..., ExternalAuthProvider]`: the Python factory
/// accepts `**kwargs`, the Rust factory receives [`ExternalProviderConfig`] and
/// may fail when a path or URL cannot be resolved.
pub type AuthProviderFactory =
    Arc<dyn Fn(&ExternalProviderConfig) -> Result<Arc<dyn ExternalAuthProvider>> + Send + Sync>;

/// Builds a model catalog from a typed configuration.
pub type ModelCatalogFactory =
    Arc<dyn Fn(&ExternalProviderConfig) -> Result<Arc<dyn ProviderModelCatalog>> + Send + Sync>;

/// One registered factory plus the spelling the caller registered it under.
struct Entry<T> {
    /// Provider spelling reported by the `list_*` functions.
    display: String,
    /// The factory itself.
    factory: T,
}

/// Both registration maps, guarded by one lock.
#[derive(Default)]
struct Registries {
    /// Auth provider factories keyed by folded provider name.
    auth: BTreeMap<String, Entry<AuthProviderFactory>>,
    /// Model catalog factories keyed by folded provider name.
    catalogs: BTreeMap<String, Entry<ModelCatalogFactory>>,
}

/// Thread-safe registry of external auth providers and model catalogs.
///
/// Use [`global`](Self::global) for the process-wide instance the free functions
/// drive, or [`new`](Self::new)/[`with_builtins`](Self::with_builtins) for an
/// isolated instance in tests or in a component that must not depend on global
/// state.
pub struct ExternalProviderRegistry {
    /// The two registration maps.
    registries: RwLock<Registries>,
}

impl ExternalProviderRegistry {
    /// Creates an empty registry.
    ///
    /// Only [`global`](Self::global) and [`with_builtins`](Self::with_builtins)
    /// pre-load the built-in `OpenAIAccount` provider.
    #[must_use]
    pub fn new() -> Self {
        Self {
            registries: RwLock::new(Registries::default()),
        }
    }

    /// Creates an empty registry and registers the built-in providers.
    ///
    /// Idempotent by construction: the registry starts empty, so the built-in
    /// insert cannot collide with a caller's registration.
    #[must_use]
    pub fn with_builtins() -> Self {
        let registry = Self::new();
        let auth: AuthProviderFactory = Arc::new(openai_account::auth_provider_factory);
        let catalog: ModelCatalogFactory = Arc::new(openai_account::model_catalog_factory);
        {
            let mut registries = registry.write();
            insert(&mut registries.auth, PROVIDER_NAME, auth);
            insert(&mut registries.catalogs, PROVIDER_NAME, catalog);
        }
        registry
    }

    /// The process-wide registry, with the built-in providers registered.
    ///
    /// The registry is created on first access and lives for the process, so
    /// every caller observes the same registrations.
    #[must_use]
    pub fn global() -> &'static Self {
        static GLOBAL: LazyLock<ExternalProviderRegistry> =
            LazyLock::new(ExternalProviderRegistry::with_builtins);
        LazyLock::force(&GLOBAL)
    }

    /// Registers an auth provider factory, returning the factory it replaced.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the provider name is empty (after
    /// trimming and folding) or when the provider is already registered and
    /// `replace` is `false`.
    pub fn register_auth_provider(
        &self,
        provider: impl AsRef<str>,
        factory: AuthProviderFactory,
        replace: bool,
    ) -> Result<Option<AuthProviderFactory>> {
        let provider = provider.as_ref();
        let key = provider_key(provider)?;
        let display = provider.trim().to_owned();
        let mut registries = self.write();
        register(&mut registries.auth, key, display, factory, "auth provider", replace)
    }

    /// Registers a model catalog factory, returning the factory it replaced.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the provider name is empty (after
    /// trimming and folding) or when the provider is already registered and
    /// `replace` is `false`.
    pub fn register_model_catalog(
        &self,
        provider: impl AsRef<str>,
        factory: ModelCatalogFactory,
        replace: bool,
    ) -> Result<Option<ModelCatalogFactory>> {
        let provider = provider.as_ref();
        let key = provider_key(provider)?;
        let display = provider.trim().to_owned();
        let mut registries = self.write();
        register(
            &mut registries.catalogs,
            key,
            display,
            factory,
            "model catalog",
            replace,
        )
    }

    /// Returns the auth provider factory registered for `provider`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an empty or unknown provider name; the
    /// unknown-provider message lists the available providers.
    pub fn get_auth_provider_factory(&self, provider: &str) -> Result<AuthProviderFactory> {
        let key = provider_key(provider)?;
        let registries = self.read();
        registries.auth.get(&key).map(|entry| entry.factory.clone()).ok_or_else(|| {
            unknown_provider(
                "auth provider",
                provider,
                registries.auth.values().map(|entry| entry.display.as_str()),
            )
        })
    }

    /// Returns the model catalog factory registered for `provider`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an empty or unknown provider name; the
    /// unknown-provider message lists the available providers.
    pub fn get_model_catalog_factory(&self, provider: &str) -> Result<ModelCatalogFactory> {
        let key = provider_key(provider)?;
        let registries = self.read();
        registries.catalogs.get(&key).map(|entry| entry.factory.clone()).ok_or_else(|| {
            unknown_provider(
                "model catalog",
                provider,
                registries.catalogs.values().map(|entry| entry.display.as_str()),
            )
        })
    }

    /// Creates a new auth provider instance for `provider`.
    ///
    /// Every call builds a fresh instance, so two creations never share state.
    ///
    /// # Errors
    ///
    /// Returns the lookup errors of [`get_auth_provider_factory`](Self::get_auth_provider_factory)
    /// and the factory's own construction errors.
    pub fn create_auth_provider(
        &self,
        provider: &str,
        config: &ExternalProviderConfig,
    ) -> Result<Arc<dyn ExternalAuthProvider>> {
        let factory = self.get_auth_provider_factory(provider)?;
        factory(config)
    }

    /// Creates a new model catalog instance for `provider`.
    ///
    /// # Errors
    ///
    /// Returns the lookup errors of [`get_model_catalog_factory`](Self::get_model_catalog_factory)
    /// and the factory's own construction errors.
    pub fn create_model_catalog(
        &self,
        provider: &str,
        config: &ExternalProviderConfig,
    ) -> Result<Arc<dyn ProviderModelCatalog>> {
        let factory = self.get_model_catalog_factory(provider)?;
        factory(config)
    }

    /// Returns the registered auth provider names, sorted.
    #[must_use]
    pub fn list_auth_providers(&self) -> Vec<String> {
        let registries = self.read();
        sorted_displays(registries.auth.values().map(|entry| entry.display.as_str()))
    }

    /// Returns the registered model catalog names, sorted.
    #[must_use]
    pub fn list_model_catalogs(&self) -> Vec<String> {
        let registries = self.read();
        sorted_displays(registries.catalogs.values().map(|entry| entry.display.as_str()))
    }

    /// Takes a read guard.
    fn read(&self) -> RwLockReadGuard<'_, Registries> {
        self.registries.read()
    }

    /// Takes a write guard.
    fn write(&self) -> RwLockWriteGuard<'_, Registries> {
        self.registries.write()
    }
}

impl Default for ExternalProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ExternalProviderRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExternalProviderRegistry")
            .field("auth_providers", &self.list_auth_providers())
            .field("model_catalogs", &self.list_model_catalogs())
            .finish()
    }
}

/// Registers an auth provider factory on the process-global registry.
///
/// # Errors
///
/// Returns the errors of
/// [`ExternalProviderRegistry::register_auth_provider`].
pub fn register_external_auth_provider(
    provider: impl AsRef<str>,
    factory: AuthProviderFactory,
    replace: bool,
) -> Result<Option<AuthProviderFactory>> {
    ExternalProviderRegistry::global().register_auth_provider(provider, factory, replace)
}

/// Returns an auth provider factory from the process-global registry.
///
/// # Errors
///
/// Returns the errors of
/// [`ExternalProviderRegistry::get_auth_provider_factory`].
pub fn get_auth_provider_factory(provider: &str) -> Result<AuthProviderFactory> {
    ExternalProviderRegistry::global().get_auth_provider_factory(provider)
}

/// Creates an auth provider from the process-global registry.
///
/// # Errors
///
/// Returns the errors of
/// [`ExternalProviderRegistry::create_auth_provider`].
pub fn create_auth_provider(
    provider: &str,
    config: &ExternalProviderConfig,
) -> Result<Arc<dyn ExternalAuthProvider>> {
    ExternalProviderRegistry::global().create_auth_provider(provider, config)
}

/// Returns the auth provider names registered on the process-global registry.
#[must_use]
pub fn list_auth_providers() -> Vec<String> {
    ExternalProviderRegistry::global().list_auth_providers()
}

/// Registers a model catalog factory on the process-global registry.
///
/// # Errors
///
/// Returns the errors of
/// [`ExternalProviderRegistry::register_model_catalog`].
pub fn register_external_model_catalog(
    provider: impl AsRef<str>,
    factory: ModelCatalogFactory,
    replace: bool,
) -> Result<Option<ModelCatalogFactory>> {
    ExternalProviderRegistry::global().register_model_catalog(provider, factory, replace)
}

/// Returns a model catalog factory from the process-global registry.
///
/// # Errors
///
/// Returns the errors of
/// [`ExternalProviderRegistry::get_model_catalog_factory`].
pub fn get_model_catalog_factory(provider: &str) -> Result<ModelCatalogFactory> {
    ExternalProviderRegistry::global().get_model_catalog_factory(provider)
}

/// Creates a model catalog from the process-global registry.
///
/// # Errors
///
/// Returns the errors of
/// [`ExternalProviderRegistry::create_model_catalog`].
pub fn create_model_catalog(
    provider: &str,
    config: &ExternalProviderConfig,
) -> Result<Arc<dyn ProviderModelCatalog>> {
    ExternalProviderRegistry::global().create_model_catalog(provider, config)
}

/// Returns the model catalog names registered on the process-global registry.
#[must_use]
pub fn list_model_catalogs() -> Vec<String> {
    ExternalProviderRegistry::global().list_model_catalogs()
}

/// Inserts an entry without validation, returning the factory it replaced.
///
/// Only the builder paths use this: the key is derived with [`folded_key`], which
/// cannot fail.
fn insert<T>(map: &mut BTreeMap<String, Entry<T>>, provider: &str, factory: T) -> Option<T> {
    let entry = Entry {
        display: provider.trim().to_owned(),
        factory,
    };
    map.insert(folded_key(provider), entry)
        .map(|previous| previous.factory)
}

/// Validates and folds a provider name into its registry key.
///
/// The key ignores case, `-`, `_`, and whitespace, so `OpenAIAccount`,
/// `openai_account`, and `openaiaccount` are the same provider.
fn provider_key(provider: &str) -> Result<String> {
    let trimmed = provider.trim();
    if trimmed.is_empty() {
        return Err(Error::validation("External provider name cannot be empty."));
    }
    let key = folded_key(trimmed);
    if key.is_empty() {
        return Err(Error::validation("External provider name cannot be empty."));
    }
    Ok(key)
}

/// Folds a provider name: lowercase and drop `-`, `_`, and whitespace.
fn folded_key(provider: &str) -> String {
    provider
        .chars()
        .filter(|character| !matches!(character, '-' | '_') && !character.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Registers `factory` under an already-validated key.
fn register<T>(
    map: &mut BTreeMap<String, Entry<T>>,
    key: String,
    display: String,
    factory: T,
    kind: &str,
    replace: bool,
) -> Result<Option<T>> {
    if let Some(existing) = map.get(&key) {
        if !replace {
            return Err(Error::validation(format!(
                "External {kind} for '{}' is already registered.",
                existing.display
            )));
        }
    }
    let entry = Entry { display, factory };
    Ok(map.insert(key, entry).map(|previous| previous.factory))
}

/// Builds the unknown-provider error, listing the available display names.
fn unknown_provider<'a>(
    kind: &str,
    provider: &str,
    available: impl Iterator<Item = &'a str>,
) -> Error {
    let available = sorted_displays(available).join(", ");
    let available = if available.is_empty() {
        "none".to_owned()
    } else {
        available
    };
    Error::validation(format!(
        "Unknown external {kind}: '{provider}'. Available providers: {available}."
    ))
}

/// Collects display names, sorted and deduplicated.
fn sorted_displays<'a>(names: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut names: Vec<String> = names.map(str::to_owned).collect();
    names.sort();
    names.dedup();
    names
}
