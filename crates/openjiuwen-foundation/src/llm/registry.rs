//! Provider registry: turns a provider name into a fully configured [`Model`].
//!
//! Ports the self-registration half of `openjiuwen/core/foundation/llm/model.py`.
//! The registry is the seam extension crates use to add transports the built-in
//! table does not know: they implement [`ModelProvider`] and hand it to
//! [`register_model_provider`], after which [`super::factory::init_model`]
//! dispatches to them exactly like a built-in.
//!
//! Every canonical name in [`super::provider::provider_names`] is registered with
//! the flavor its profile declares, so `openai` (and every other
//! OpenAI-compatible vendor) builds an
//! [`OpenAiCompatibleClient`] and
//! `anthropic` builds an [`AnthropicClient`].
//! Registration performs no I/O: a provider is only asked to build a client when
//! [`ModelProvider::create`] is called.
//!
//! ```
//! use openjiuwen_foundation::llm::ModelProviderRegistry;
//!
//! let registry = ModelProviderRegistry::global();
//! assert_eq!(registry.resolve("DashScope").map(|provider| provider.name().to_owned()), Some("qwen".to_owned()));
//! assert!(registry.resolve("acme").is_none());
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, LazyLock, PoisonError, RwLock};

use openjiuwen_common::error::Result;

use super::anthropic::AnthropicClient;
use super::config::{ModelClientConfig, ModelRequestConfig};
use super::model::Model;
use super::openai::OpenAiCompatibleClient;
use super::provider::{normalize_provider_name, provider_names, provider_profile, ProviderFlavor};

/// Builds model clients for one provider.
///
/// Extension crates implement this trait to plug a transport the built-in table
/// does not know into [`super::factory::init_model`]. Implementations must be
/// cheap and side-effect free: [`create`](Self::create) runs on every factory
/// call and must not perform network I/O or block on a lock held elsewhere.
pub trait ModelProvider: Send + Sync {
    /// Canonical name this provider is registered under.
    ///
    /// For built-ins this is the [`ProviderProfile::name`](super::provider::ProviderProfile::name)
    /// of the table entry, for example `"openai"` or `"anthropic"`.
    fn name(&self) -> &str;

    /// Additional spellings that [`ModelProviderRegistry::resolve`] accepts.
    ///
    /// Entries are normalized with
    /// [`normalize_provider_name`]
    /// before comparison, so `"acme-ai"`, `"Acme_AI"`, and `"acmeai"` all match.
    /// The default is "no extra spellings"; the built-ins rely on
    /// [`normalize_provider_name`] for
    /// vendor aliases instead of listing them here.
    fn aliases(&self) -> &[&str] {
        &[]
    }

    /// Default API base advertised by this provider, if it has one.
    ///
    /// Informational: the endpoint actually used is the one in the resolved
    /// [`ModelClientConfig`]. Built-ins return their profile's
    /// [`ProviderProfile::default_api_base`](super::provider::ProviderProfile::default_api_base),
    /// which is `None` for a deployment such as `pangu` that must be given one.
    fn default_api_base(&self) -> Option<&str> {
        None
    }

    /// Builds a fully configured client for `client` and `request`.
    ///
    /// The returned model owns its configuration, so it must not re-read the
    /// environment. The configuration handed in is already resolved by the
    /// factory for built-in names; extension providers receive their own config
    /// back with only the provider name normalized, and may resolve it as they
    /// see fit (see [`super::factory::init_model`]).
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Config`] or
    /// [`openjiuwen_common::error::Error::Unsupported`] when the configuration
    /// cannot produce a usable client: a missing endpoint, a missing credential,
    /// or an unusable HTTP transport.
    fn create(
        &self,
        client: &ModelClientConfig,
        request: &ModelRequestConfig,
    ) -> Result<Arc<dyn Model>>;
}

/// Thread-safe map from canonical provider name to [`ModelProvider`].
///
/// Use [`global`](Self::global) for the process-wide registry the factory
/// consults, and [`new`](Self::new) for an isolated instance in tests or in a
/// component that must not depend on global state.
///
/// ```
/// use openjiuwen_foundation::llm::ModelProviderRegistry;
///
/// let registry = ModelProviderRegistry::new();
/// assert!(registry.names().is_empty());
/// assert!(registry.resolve("openai").is_none());
/// ```
pub struct ModelProviderRegistry {
    /// Providers keyed by canonical name.
    providers: RwLock<BTreeMap<String, Arc<dyn ModelProvider>>>,
}

impl ModelProviderRegistry {
    /// Creates an empty registry.
    ///
    /// A local registry starts empty; only [`global`](Self::global) pre-loads the
    /// built-in providers.
    #[must_use]
    pub fn new() -> Self {
        Self {
            providers: RwLock::new(BTreeMap::new()),
        }
    }

    /// The process-wide registry, with the built-in providers registered.
    ///
    /// The registry is created on first access and lives for the process, so
    /// every caller observes the same registrations. It is safe to call from
    /// any thread; reads and writes take a short-lived lock.
    #[must_use]
    pub fn global() -> &'static Self {
        static GLOBAL: LazyLock<ModelProviderRegistry> = LazyLock::new(|| {
            let registry = ModelProviderRegistry::new();
            registry.register_builtins();
            registry
        });
        LazyLock::force(&GLOBAL)
    }

    /// Registers `provider`, returning the provider it replaced.
    ///
    /// Entries are keyed by [`ModelProvider::name`], so registering a second
    /// provider with the same name replaces the first: that is how an extension
    /// crate overrides a built-in (or another extension) for the rest of the
    /// process. Returns the previous provider so a caller can restore it.
    pub fn register(&self, provider: Arc<dyn ModelProvider>) -> Option<Arc<dyn ModelProvider>> {
        let name = provider.name().to_owned();
        self.write().insert(name, provider)
    }

    /// Looks up the provider registered under exactly `name`.
    ///
    /// No normalization happens here: built-ins are keyed by canonical name
    /// (`"qwen"`, not `"DashScope"`). Use [`resolve`](Self::resolve) for
    /// user-supplied names.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn ModelProvider>> {
        self.read().get(name).cloned()
    }

    /// Looks up `name` after normalization and alias resolution.
    ///
    /// The name is normalized with
    /// [`normalize_provider_name`]
    /// (case, `-`, `_`, and whitespace insensitive, vendor aliases folded), then
    /// every registered provider's [`aliases`](ModelProvider::aliases) are
    /// compared in the same normalized form, and finally the result is looked up
    /// with [`get`](Self::get). Aliases therefore win over an exact-name entry,
    /// which lets an extension crate claim a vendor spelling without renaming its
    /// own provider.
    ///
    /// Returns `None` for a name that is neither registered nor an alias of a
    /// registered provider.
    #[must_use]
    pub fn resolve(&self, name: &str) -> Option<Arc<dyn ModelProvider>> {
        let canonical = normalize_provider_name(name);
        let by_alias = self.snapshot().into_iter().find(|provider| {
            provider
                .aliases()
                .iter()
                .any(|alias| normalize_provider_name(alias) == canonical)
        });
        by_alias.or_else(|| self.get(&canonical))
    }

    /// Every registered name, sorted.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.read().keys().cloned().collect()
    }

    /// Registers every built-in provider in the static provider table.
    ///
    /// Idempotent: each entry replaces a same-named registration, so calling it
    /// again neither duplicates providers nor fails. One provider is created per
    /// canonical name, with the flavor
    /// [`provider_profile`](super::provider::provider_profile) declares.
    fn register_builtins(&self) {
        for name in provider_names() {
            let Some(profile) = provider_profile(name) else {
                continue;
            };
            let provider: Arc<dyn ModelProvider> = match profile.flavor {
                ProviderFlavor::OpenAiCompatible => Arc::new(OpenAiProvider {
                    name: profile.name,
                    default_api_base: profile.default_api_base,
                }),
                ProviderFlavor::Anthropic => Arc::new(AnthropicProvider {
                    name: profile.name,
                    default_api_base: profile.default_api_base,
                }),
            };
            self.register(provider);
        }
    }

    /// Reads the guard, recovering from a poisoned lock.
    fn read(&self) -> std::sync::RwLockReadGuard<'_, BTreeMap<String, Arc<dyn ModelProvider>>> {
        self.providers
            .read()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Writes the guard, recovering from a poisoned lock.
    fn write(&self) -> std::sync::RwLockWriteGuard<'_, BTreeMap<String, Arc<dyn ModelProvider>>> {
        self.providers
            .write()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Clones the registered providers so callers can inspect them lock-free.
    fn snapshot(&self) -> Vec<Arc<dyn ModelProvider>> {
        self.read().values().cloned().collect()
    }
}

impl Default for ModelProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ModelProviderRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelProviderRegistry")
            .field("providers", &self.names())
            .finish()
    }
}

/// Registers a provider in the process-wide registry.
///
/// Registration is process-global and replaces an existing entry of the same
/// name, so an extension crate can override a built-in for the whole process.
/// Pairs with [`ModelProviderRegistry::resolve`]: the factory sees the new
/// provider immediately.
pub fn register_model_provider(provider: Arc<dyn ModelProvider>) {
    let name = provider.name().to_owned();
    if let Some(previous) = ModelProviderRegistry::global().register(provider) {
        tracing::debug!(
            provider = %name,
            replaced = %previous.name(),
            "replaced a registered model provider"
        );
    }
}

/// Registers every built-in provider in the process-wide registry.
///
/// [`ModelProviderRegistry::global`] already does this on first access; the
/// explicit entry point exists for callers that want the work done eagerly (and
/// for tests). Registration is idempotent: re-running it replaces each entry
/// with an equivalent one instead of duplicating names.
pub fn register_builtin_providers() {
    ModelProviderRegistry::global().register_builtins();
}

/// Built-in provider speaking the OpenAI-compatible chat-completions API.
#[derive(Debug, Clone, Copy)]
struct OpenAiProvider {
    /// Canonical provider name from the static table.
    name: &'static str,
    /// Default endpoint advertised by the table.
    default_api_base: Option<&'static str>,
}

impl ModelProvider for OpenAiProvider {
    fn name(&self) -> &str {
        self.name
    }

    fn default_api_base(&self) -> Option<&str> {
        self.default_api_base
    }

    fn create(
        &self,
        client: &ModelClientConfig,
        request: &ModelRequestConfig,
    ) -> Result<Arc<dyn Model>> {
        let client = with_provider(client, self.name);
        let model = OpenAiCompatibleClient::new(client, request.clone())?;
        Ok(Arc::new(model))
    }
}

/// Built-in provider speaking the Anthropic Messages API.
#[derive(Debug, Clone, Copy)]
struct AnthropicProvider {
    /// Canonical provider name from the static table.
    name: &'static str,
    /// Default endpoint advertised by the table.
    default_api_base: Option<&'static str>,
}

impl ModelProvider for AnthropicProvider {
    fn name(&self) -> &str {
        self.name
    }

    fn default_api_base(&self) -> Option<&str> {
        self.default_api_base
    }

    fn create(
        &self,
        client: &ModelClientConfig,
        request: &ModelRequestConfig,
    ) -> Result<Arc<dyn Model>> {
        let client = with_provider(client, self.name);
        let model = AnthropicClient::new(client, request.clone())?;
        Ok(Arc::new(model))
    }
}

/// Fills in a blank [`ModelClientConfig::provider`] from the registered name.
///
/// A caller that reaches a provider through the registry knows which provider it
/// asked for, so a config with an empty provider field must not silently fall
/// back to the client constructor's default (`openai` even for `qwen`). A
/// non-blank provider is left untouched: both constructors normalize vendor
/// spellings themselves.
fn with_provider(client: &ModelClientConfig, name: &str) -> ModelClientConfig {
    let mut client = client.clone();
    if client.provider.trim().is_empty() {
        name.clone_into(&mut client.provider);
    }
    client
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::Error;

    /// A provider that records nothing and cannot build a model.
    #[derive(Debug)]
    struct FakeProvider {
        name: &'static str,
        aliases: &'static [&'static str],
        default_api_base: Option<&'static str>,
    }

    impl FakeProvider {
        const fn new(name: &'static str) -> Self {
            Self {
                name,
                aliases: &[],
                default_api_base: None,
            }
        }
    }

    impl ModelProvider for FakeProvider {
        fn name(&self) -> &str {
            self.name
        }

        fn aliases(&self) -> &[&str] {
            self.aliases
        }

        fn default_api_base(&self) -> Option<&str> {
            self.default_api_base
        }

        fn create(
            &self,
            _client: &ModelClientConfig,
            _request: &ModelRequestConfig,
        ) -> Result<Arc<dyn Model>> {
            Err(Error::unsupported(format!(
                "`{}` is a registry test double",
                self.name
            )))
        }
    }

    fn registered_provider(name: &str) -> Arc<dyn ModelProvider> {
        ModelProviderRegistry::global()
            .resolve(name)
            .unwrap_or_else(|| panic!("{name} is registered"))
    }

    #[test]
    fn global_registers_every_builtin_provider() {
        let registry = ModelProviderRegistry::global();
        for name in provider_names() {
            let provider = registry
                .get(name)
                .unwrap_or_else(|| panic!("{name} is built in"));
            assert_eq!(provider.name(), *name);

            let resolved = registered_provider(name);
            assert_eq!(resolved.name(), *name);
        }
        assert_eq!(
            registry.names().len(),
            provider_names().len(),
            "the global registry holds exactly the built-ins"
        );
    }

    #[test]
    fn builtins_expose_their_profile_api_base() {
        let cases = [
            (
                "qwen",
                Some("https://dashscope.aliyuncs.com/compatible-mode/v1"),
            ),
            ("anthropic", Some("https://api.anthropic.com")),
            ("ollama", Some("http://localhost:11434/v1")),
            ("pangu", None),
        ];
        for (name, expected) in cases {
            assert_eq!(
                registered_provider(name).default_api_base(),
                expected,
                "default api base of {name}"
            );
        }
    }

    #[test]
    fn resolve_normalizes_case_separators_and_vendor_aliases() {
        let cases = [
            ("openai", "openai"),
            ("  Open-AI ", "openai"),
            ("DashScope", "qwen"),
            ("dash_scope", "qwen"),
            ("claude", "anthropic"),
            ("Claude", "anthropic"),
            ("kimi", "moonshot"),
            ("LM_Studio", "lmstudio"),
            ("silicon_flow", "siliconflow"),
            ("DeepSeek", "deepseek"),
            ("ollama", "ollama"),
        ];
        for (input, canonical) in cases {
            let provider = ModelProviderRegistry::global()
                .resolve(input)
                .unwrap_or_else(|| panic!("{input} resolves"));
            assert_eq!(provider.name(), canonical, "resolving {input}");
        }
    }

    #[test]
    fn unknown_names_do_not_resolve() {
        let registry = ModelProviderRegistry::global();
        for name in ["acme", "my-custom-provider", "", "  ", "openai-compatible"] {
            assert!(
                registry.resolve(name).is_none(),
                "{name:?} must not resolve"
            );
        }
    }

    #[test]
    fn names_are_sorted_and_complete() {
        let mut expected: Vec<String> = provider_names()
            .iter()
            .map(|name| (*name).to_owned())
            .collect();
        expected.sort();
        assert_eq!(ModelProviderRegistry::global().names(), expected);
    }

    #[test]
    fn new_registry_is_empty_and_registration_returns_the_previous_provider() {
        let registry = ModelProviderRegistry::new();
        assert_eq!(registry.names(), Vec::<String>::new());
        assert!(registry.resolve("acme").is_none());

        let first: Arc<dyn ModelProvider> = Arc::new(FakeProvider::new("acme"));
        assert!(registry.register(first.clone()).is_none());
        assert!(Arc::ptr_eq(
            &registry.get("acme").expect("acme is registered"),
            &first
        ));

        let second: Arc<dyn ModelProvider> = Arc::new(FakeProvider::new("acme"));
        let previous = registry
            .register(second.clone())
            .expect("a same-name registration is replaced");
        assert!(Arc::ptr_eq(&previous, &first));
        assert!(Arc::ptr_eq(
            &registry.get("acme").expect("acme is still registered"),
            &second
        ));
        assert_eq!(registry.names(), vec!["acme".to_owned()]);
    }

    #[test]
    fn get_is_exact_and_resolve_normalizes() {
        let registry = ModelProviderRegistry::new();
        registry.register(Arc::new(FakeProvider::new("acme")));

        assert!(registry.get("ACME").is_none(), "get does not normalize");
        assert!(registry.get("Acme-AI").is_none());
        assert_eq!(
            registry
                .resolve("ACME")
                .map(|provider| provider.name().to_owned()),
            Some("acme".to_owned())
        );
    }

    #[test]
    fn resolve_consults_registered_aliases() {
        let registry = ModelProviderRegistry::new();
        registry.register(Arc::new(FakeProvider {
            name: "acme",
            aliases: &["acme-ai", "Acme_Cloud"],
            default_api_base: None,
        }));

        for alias in ["acme-ai", "ACME-AI", "acme_ai", "acmeai", "Acme Cloud"] {
            let provider = registry
                .resolve(alias)
                .unwrap_or_else(|| panic!("{alias} is an alias of acme"));
            assert_eq!(provider.name(), "acme");
        }
        assert!(registry.resolve("acme-enterprise").is_none());
    }

    #[test]
    fn global_is_a_stable_singleton() {
        let first = ModelProviderRegistry::global();
        let second = ModelProviderRegistry::global();
        assert!(std::ptr::eq(first, second));
        assert_ne!(first.names(), Vec::<String>::new());
    }

    #[test]
    fn registering_builtins_is_idempotent() {
        let registry = ModelProviderRegistry::global();
        let before = registry.names();
        register_builtin_providers();
        register_builtin_providers();
        assert_eq!(registry.names(), before);
        assert_eq!(registry.names().len(), provider_names().len());
        assert_eq!(registered_provider("anthropic").name(), "anthropic");
    }

    #[test]
    fn register_model_provider_replaces_by_name() {
        let registry = ModelProviderRegistry::global();
        let before = registry.names();

        register_model_provider(Arc::new(OpenAiProvider {
            name: "openai",
            default_api_base: Some("https://api.openai.com/v1"),
        }));

        assert_eq!(registry.names(), before, "replacing adds no name");
        assert_eq!(registered_provider("openai").name(), "openai");
        assert_eq!(registered_provider("DashScope").name(), "qwen");
    }

    #[test]
    fn default_registry_is_empty_and_debug_lists_names() {
        let registry = ModelProviderRegistry::default();
        assert_eq!(registry.names(), Vec::<String>::new());
        assert_eq!(
            format!("{registry:?}"),
            "ModelProviderRegistry { providers: [] }"
        );

        registry.register(Arc::new(FakeProvider::new("acme")));
        assert_eq!(
            format!("{registry:?}"),
            "ModelProviderRegistry { providers: [\"acme\"] }"
        );
    }
}
