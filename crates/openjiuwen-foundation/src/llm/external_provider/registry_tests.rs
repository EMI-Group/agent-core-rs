//! Registry tests for the external provider seam.
//!
//! Every test builds an isolated [`ExternalProviderRegistry`]; the process-global
//! registry is only *read* (and only by the isolation test), so no test depends on
//! registration order or leaks a registration into another test.
//! Ports the behavior of `openjiuwen/extensions/external_provider/registry.py`.
//!
//! No test performs network I/O: the built-in credential manager and model catalog
//! resolve every path from the supplied [`ExternalProviderConfig`] and are only
//! constructed, never called against an endpoint.

use std::path::PathBuf;
use std::sync::Arc;

use openjiuwen_common::error::{Error, ErrorKind, Result};

use super::base::{
    ExternalAuthProvider, ExternalProviderConfig, ModelCatalogRequest, ProviderModelCatalog,
};
use super::openai_account::{DEFAULT_MODELS, MISSING_AUTH_CODE, PROVIDER_NAME};
use super::registry::{
    create_auth_provider, list_auth_providers, list_model_catalogs, AuthProviderFactory,
    ExternalProviderRegistry, ModelCatalogFactory,
};

/// Builds a provider configuration whose auth store and model cache both live in
/// `directory`, so nothing resolves from the ambient environment.
fn config_in(directory: &tempfile::TempDir) -> ExternalProviderConfig {
    ExternalProviderConfig::new()
        .with_auth_path(directory.path().join("auth.json"))
        .with_cache_path(directory.path().join("openai_account_models_cache.json"))
}

/// Returns the auth-store path `config_in` configured.
fn auth_path_in(directory: &tempfile::TempDir) -> PathBuf {
    directory.path().join("auth.json")
}

/// Builds an auth provider factory that always fails with `message`.
///
/// A failing factory makes "which factory is registered" observable through
/// `create_auth_provider` without needing a real provider implementation.
fn failing_auth_factory(message: &'static str) -> AuthProviderFactory {
    Arc::new(
        move |_config: &ExternalProviderConfig| -> Result<Arc<dyn ExternalAuthProvider>> {
            Err(Error::validation(message))
        },
    )
}

/// Builds a model catalog factory that always fails with `message`.
fn failing_catalog_factory(message: &'static str) -> ModelCatalogFactory {
    Arc::new(
        move |_config: &ExternalProviderConfig| -> Result<Arc<dyn ProviderModelCatalog>> {
            Err(Error::validation(message))
        },
    )
}

/// Returns the provider names a list contains, for assertion messages.
fn listed(names: &[String]) -> String {
    names.join(", ")
}

// --- built-ins --------------------------------------------------------------

#[tokio::test]
async fn with_builtins_lists_and_creates_the_openai_account_provider() {
    let registry = ExternalProviderRegistry::with_builtins();
    assert_eq!(
        registry.list_auth_providers(),
        vec![PROVIDER_NAME.to_owned()],
        "the built-in provider is the only auth provider"
    );
    assert_eq!(
        registry.list_model_catalogs(),
        vec![PROVIDER_NAME.to_owned()],
        "the built-in provider is the only model catalog"
    );

    let directory = tempfile::tempdir().expect("a temp directory");
    let config = config_in(&directory);

    let auth = registry
        .create_auth_provider(PROVIDER_NAME, &config)
        .expect("the built-in auth provider");
    let status = auth.status().await.expect("the status");
    assert!(
        !status.authenticated,
        "a fresh temp store holds no credential"
    );
    assert_eq!(
        status.auth_path.as_deref(),
        Some(auth_path_in(&directory).as_path()),
        "the configured auth path is reported"
    );
    assert_eq!(status.error.as_deref(), Some(MISSING_AUTH_CODE));

    let catalog = registry
        .create_model_catalog(PROVIDER_NAME, &config)
        .expect("the built-in model catalog");
    let model_ids = catalog
        .list_model_ids(&ModelCatalogRequest::new())
        .await
        .expect("the defaults answer without a credential");
    assert_eq!(
        model_ids,
        DEFAULT_MODELS.map(str::to_owned).to_vec(),
        "an empty cache and no token answer with the built-in defaults"
    );
}

#[test]
fn provider_names_fold_case_and_separators() {
    let registry = ExternalProviderRegistry::with_builtins();
    let canonical_auth = registry
        .get_auth_provider_factory(PROVIDER_NAME)
        .expect("the built-in auth factory");
    let canonical_catalog = registry
        .get_model_catalog_factory(PROVIDER_NAME)
        .expect("the built-in model catalog factory");

    for spelling in [
        "OpenAIAccount",
        "openai_account",
        "openai-account",
        "openaiaccount",
        "  OpenAI Account ",
    ] {
        let auth = registry
            .get_auth_provider_factory(spelling)
            .unwrap_or_else(|error| panic!("{spelling}: {error}"));
        assert!(
            Arc::ptr_eq(&auth, &canonical_auth),
            "{spelling} must resolve to the built-in auth factory"
        );
        let catalog = registry
            .get_model_catalog_factory(spelling)
            .unwrap_or_else(|error| panic!("{spelling}: {error}"));
        assert!(
            Arc::ptr_eq(&catalog, &canonical_catalog),
            "{spelling} must resolve to the built-in model catalog factory"
        );
    }
}

#[tokio::test]
async fn every_creation_builds_a_fresh_instance() {
    let registry = ExternalProviderRegistry::with_builtins();
    let directory = tempfile::tempdir().expect("a temp directory");
    let config = config_in(&directory);

    let first_auth = registry
        .create_auth_provider(PROVIDER_NAME, &config)
        .expect("the first auth provider");
    let second_auth = registry
        .create_auth_provider(PROVIDER_NAME, &config)
        .expect("the second auth provider");
    assert!(
        !Arc::ptr_eq(&first_auth, &second_auth),
        "each creation must build an independent manager"
    );

    let first_catalog = registry
        .create_model_catalog(PROVIDER_NAME, &config)
        .expect("the first model catalog");
    let second_catalog = registry
        .create_model_catalog(PROVIDER_NAME, &config)
        .expect("the second model catalog");
    assert!(
        !Arc::ptr_eq(&first_catalog, &second_catalog),
        "each creation must build an independent catalog"
    );
}

// --- lookup failures --------------------------------------------------------

#[test]
fn an_unknown_catalog_lists_the_available_providers() {
    let registry = ExternalProviderRegistry::with_builtins();
    let directory = tempfile::tempdir().expect("a temp directory");

    let error = registry
        .create_model_catalog("MissingProvider", &config_in(&directory))
        .err()
        .expect("the provider is not registered");

    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(
        message.contains("Unknown external model catalog"),
        "the message names the failed lookup: {message}"
    );
    assert!(
        message.contains(PROVIDER_NAME),
        "the message lists the available providers: {message}"
    );
}

#[test]
fn an_unknown_auth_provider_lists_the_available_providers() {
    let registry = ExternalProviderRegistry::with_builtins();

    let error = registry
        .get_auth_provider_factory("MissingProvider")
        .err()
        .expect("the provider is not registered");

    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(
        message.contains("Unknown external auth provider"),
        "message: {message}"
    );
    assert!(message.contains(PROVIDER_NAME), "message: {message}");
}

#[test]
fn empty_provider_names_are_rejected() {
    let registry = ExternalProviderRegistry::with_builtins();
    let config = ExternalProviderConfig::new();

    for spelling in ["", "   ", "\t\n", "-_-"] {
        let label = format!("{spelling:?}");

        let error = registry
            .create_auth_provider(spelling, &config)
            .err()
            .expect("an empty name has no provider");
        assert_eq!(error.kind(), ErrorKind::Validation, "{label}");
        assert!(
            error.to_string().contains("cannot be empty"),
            "{label}: {error}"
        );

        let error = registry
            .create_model_catalog(spelling, &config)
            .err()
            .expect("an empty name has no catalog");
        assert_eq!(error.kind(), ErrorKind::Validation, "{label}");
        assert!(
            error.to_string().contains("cannot be empty"),
            "{label}: {error}"
        );

        let error = registry
            .register_auth_provider(spelling, failing_auth_factory("unused"), false)
            .err()
            .expect("an empty name cannot be registered");
        assert_eq!(error.kind(), ErrorKind::Validation, "{label}");
        assert!(
            error.to_string().contains("cannot be empty"),
            "{label}: {error}"
        );
    }
}

// --- registration rules -----------------------------------------------------

#[tokio::test]
async fn a_duplicate_registration_requires_replace() {
    let registry = ExternalProviderRegistry::new();
    let config = ExternalProviderConfig::new();
    let original = failing_auth_factory("original auth factory");
    assert!(
        registry
            .register_auth_provider("Custom", original.clone(), false)
            .expect("the first registration")
            .is_none(),
        "the first registration replaces nothing"
    );

    // Python's key is `strip().lower()`, so `custom` collides with `Custom`.
    let error = registry
        .register_auth_provider("custom", failing_auth_factory("second auth factory"), false)
        .err()
        .expect("a duplicate registration is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("already registered"), "message: {message}");
    assert!(
        message.contains("Custom"),
        "the message names the registered provider: {message}"
    );

    let replacement = failing_auth_factory("replacement auth factory");
    let previous = registry
        .register_auth_provider("  CUSTOM  ", replacement, true)
        .expect("replace is allowed")
        .expect("the replaced factory is returned");
    assert!(
        Arc::ptr_eq(&previous, &original),
        "the previous factory is handed back so a caller can restore it"
    );

    // The new factory wins: its error is the one creation reports.
    let error = registry
        .create_auth_provider("custom", &config)
        .err()
        .expect("the replacement factory fails");
    assert!(
        error.to_string().contains("replacement auth factory"),
        "the last registration wins: {error}"
    );
}

#[tokio::test]
async fn a_duplicate_catalog_registration_requires_replace() {
    let registry = ExternalProviderRegistry::new();
    let config = ExternalProviderConfig::new();
    let original = failing_catalog_factory("original catalog factory");
    registry
        .register_model_catalog("CustomCatalog", original.clone(), false)
        .expect("the first registration");

    let error = registry
        .register_model_catalog("customcatalog", failing_catalog_factory("second"), false)
        .err()
        .expect("a duplicate registration is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("already registered"), "message: {message}");
    assert!(
        message.contains("CustomCatalog"),
        "the message names the registered provider: {message}"
    );

    let previous = registry
        .register_model_catalog("custom_catalog", failing_catalog_factory("new"), true)
        .expect("replace is allowed")
        .expect("the replaced factory is returned");
    assert!(Arc::ptr_eq(&previous, &original));

    let error = registry
        .create_model_catalog("CustomCatalog", &config)
        .err()
        .expect("the replacement factory fails");
    assert!(
        error.to_string().contains("new"),
        "the last registration wins: {error}"
    );
}

/// The registration names a local registry reports, in sorted order.
#[tokio::test]
async fn a_local_registry_never_leaks_into_the_global_one() {
    let local = ExternalProviderRegistry::with_builtins();
    local
        .register_auth_provider("LocalOnlyProvider", failing_auth_factory("local"), false)
        .expect("the local auth registration");
    local
        .register_model_catalog("LocalOnlyCatalog", failing_catalog_factory("local"), false)
        .expect("the local catalog registration");

    assert!(
        local
            .list_auth_providers()
            .contains(&"LocalOnlyProvider".to_owned()),
        "the local registry lists its own registration: {}",
        listed(&local.list_auth_providers())
    );

    assert!(
        list_auth_providers().contains(&PROVIDER_NAME.to_owned()),
        "the global registry keeps the built-in provider: {}",
        listed(&list_auth_providers())
    );
    assert!(
        !list_auth_providers().contains(&"LocalOnlyProvider".to_owned()),
        "a local registration stays local: {}",
        listed(&list_auth_providers())
    );
    assert!(
        list_model_catalogs().contains(&PROVIDER_NAME.to_owned()),
        "the global registry keeps the built-in catalog: {}",
        listed(&list_model_catalogs())
    );
    assert!(
        !list_model_catalogs().contains(&"LocalOnlyCatalog".to_owned()),
        "a local registration stays local: {}",
        listed(&list_model_catalogs())
    );

    let error = ExternalProviderRegistry::global()
        .get_auth_provider_factory("LocalOnlyProvider")
        .err()
        .expect("the global registry never sees the local registration");
    assert_eq!(error.kind(), ErrorKind::Validation);

    let directory = tempfile::tempdir().expect("a temp directory");
    let config = config_in(&directory);
    assert!(
        create_auth_provider("openai_account", &config).is_ok(),
        "the free functions still reach the built-in provider"
    );
}

#[test]
fn the_free_functions_list_the_builtin_provider() {
    let auth = list_auth_providers();
    assert!(
        auth.contains(&PROVIDER_NAME.to_owned()),
        "auth providers: {}",
        listed(&auth)
    );
    let catalogs = list_model_catalogs();
    assert!(
        catalogs.contains(&PROVIDER_NAME.to_owned()),
        "model catalogs: {}",
        listed(&catalogs)
    );
}
