//! Provider factory: the `init_model` entry point used by every layer above.
//!
//! Ports `init_model` from `openjiuwen/core/foundation/llm/model.py`. The factory
//! turns a [`ModelClientConfig`] plus a [`ModelRequestConfig`] into an
//! `Arc<dyn Model>` by asking the process-wide
//! [`ModelProviderRegistry`] for the provider that owns the configured name. It
//! performs no I/O beyond constructing the HTTP transport; the returned client is
//! ready for [`Model::invoke`] and [`Model::stream`].
//!
//! Two entry points exist:
//!
//! - [`init_model`] takes both configs and honours everything in them, which is
//!   what a runner or an application with explicit settings uses.
//! - [`init_model_with`] mirrors the Python convenience `init_model(provider,
//!   model_name)` helper, filling the configs from the environment with the
//!   Python factory's defaults.

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::config::env;
use openjiuwen_common::error::{Error, Result};

use super::config::{ModelClientConfig, ModelRequestConfig};
use super::model::Model;
use super::provider::{normalize_provider_name, provider_profile};
use super::registry::ModelProviderRegistry;

/// Total request timeout applied by [`init_model_with`] (Python factory default).
const FACTORY_TIMEOUT: Duration = Duration::from_secs(60);

/// TLS verification applied by [`init_model_with`] (Python factory default).
const FACTORY_VERIFY_SSL: bool = false;

/// Retry budget applied by [`init_model_with`] (Python factory default).
const FACTORY_MAX_RETRIES: u32 = 3;

/// Builds a model client for the provider and model named in the configs.
///
/// The provider comes from [`ModelClientConfig::provider`] when it is non-blank
/// and from the `MODEL_PROVIDER` environment variable (default `"openai"`)
/// otherwise. The name is resolved through the process-wide
/// [`ModelProviderRegistry`], so built-in providers and providers registered by
/// extension crates both work.
///
/// Resolution differs by provider kind:
///
/// - **Built-in names** (anything in the static provider table, including vendor
///   aliases such as `dashscope`) receive a config already resolved with
///   [`ModelClientConfig::resolve_for`]: environment fallbacks for the API key and
///   base URL, the provider's default endpoint, its static headers, and its
///   credential requirements are applied exactly once, before the provider builds
///   the client.
/// - **Extension names** (registered by another crate and absent from the static
///   table) receive the config with only
///   [`provider`](ModelClientConfig::provider) set to the provider's canonical
///   name. No environment fallback is applied, because the factory cannot know
///   which variables an unknown vendor honours; the provider resolves its own
///   config.
///
/// # Errors
///
/// Returns [`Error::Unsupported`] when no registered provider claims the name (the
/// message names the provider and lists the known canonical names), and whatever
/// the provider returns from [`ModelProvider::create`](super::registry::ModelProvider::create) — typically
/// [`Error::Config`] for a missing endpoint, a missing credential, or a provider
/// that speaks a different wire protocol.
///
/// # Examples
///
/// ```no_run
/// use openjiuwen_foundation::llm::{init_model, ModelClientConfig, ModelRequestConfig};
///
/// # async fn example() -> openjiuwen_common::error::Result<()> {
/// let client = ModelClientConfig::from_env();
/// let request = ModelRequestConfig::from_env().with_model_name("gpt-4o-mini");
/// let _model = init_model(client, request).await?;
/// # Ok(())
/// # }
/// ```
pub async fn init_model(
    client: ModelClientConfig,
    request: ModelRequestConfig,
) -> Result<Arc<dyn Model>> {
    init_model_in(ModelProviderRegistry::global(), client, request).await
}

/// Convenience constructor that fills the configs from the environment.
///
/// Equivalent to the Python `init_model(provider, model_name)` helper: the client
/// config comes from [`ModelClientConfig::from_env`] with `provider` set to
/// `provider` and the Python factory's defaults — a 60 s timeout, TLS
/// verification disabled, and 3 retries — while the request config comes from
/// [`ModelRequestConfig::from_env`] with `model_name` set to `model_name`.
///
/// Those defaults deliberately differ from [`ModelClientConfig::default`], which
/// keeps the stricter Python dataclass values (360 s, TLS verification on, no
/// retries): this entry point is the permissive "just try my endpoint" helper.
/// Both configs still read the environment, so `API_KEY`, `API_BASE`, and other
/// `OPENJIUWEN_*`/`OPENAI_*` variables apply.
///
/// `provider` is dispatched like [`init_model`] does, so an unknown provider is an
/// [`Error::Unsupported`] naming the known ones.
///
/// # Errors
///
/// Propagates every error of [`init_model`].
///
/// # Examples
///
/// ```no_run
/// use openjiuwen_foundation::llm::init_model_with;
///
/// # async fn example() -> openjiuwen_common::error::Result<()> {
/// let _model = init_model_with("openai", "gpt-4o-mini").await?;
/// # Ok(())
/// # }
/// ```
pub async fn init_model_with(
    provider: impl Into<String>,
    model_name: impl Into<String>,
) -> Result<Arc<dyn Model>> {
    let client = factory_client_config(provider);
    let request = ModelRequestConfig::from_env().with_model_name(model_name);
    init_model(client, request).await
}

/// Runs the factory against `registry`.
///
/// Split out from [`init_model`] so tests can exercise dispatch and config
/// resolution against an isolated registry without touching the process-wide one.
async fn init_model_in(
    registry: &ModelProviderRegistry,
    client: ModelClientConfig,
    request: ModelRequestConfig,
) -> Result<Arc<dyn Model>> {
    let requested = requested_provider(&client);
    let provider = registry.resolve(&requested).ok_or_else(|| {
        Error::unsupported(format!(
            "unsupported model provider `{requested}`; known providers: {}",
            registry.names().join(", ")
        ))
    })?;

    let canonical = normalize_provider_name(provider.name());
    let client = if provider_profile(&canonical).is_some() {
        client.resolve_for(&canonical)?
    } else {
        let mut client = client;
        client.provider = canonical;
        client
    };

    provider.create(&client, &request)
}

/// The provider name to dispatch on: the configured one, else the env default.
fn requested_provider(client: &ModelClientConfig) -> String {
    let provider = client.provider.trim();
    if provider.is_empty() {
        env::model_provider()
    } else {
        provider.to_owned()
    }
}

/// The client config [`init_model_with`] builds: environment values plus the
/// Python factory defaults (60 s timeout, no TLS verification, 3 retries).
fn factory_client_config(provider: impl Into<String>) -> ModelClientConfig {
    let mut client = ModelClientConfig::from_env();
    client.provider = provider.into();
    client.timeout = Some(FACTORY_TIMEOUT);
    client.verify_ssl = FACTORY_VERIFY_SSL;
    client.max_retries = FACTORY_MAX_RETRIES;
    client
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    use async_trait::async_trait;
    use futures::stream::BoxStream;

    use crate::llm::model::{ChatResponse, StreamChunk};
    use crate::llm::openai::OpenAiCompatibleClient;
    use crate::llm::registry::ModelProvider;
    use crate::prompt::BaseMessage;

    use openjiuwen_common::error::ErrorKind;

    /// A model that never talks to a provider, used by extension-provider tests.
    #[derive(Debug)]
    struct FakeModel {
        provider: String,
        model_name: String,
    }

    #[async_trait]
    impl Model for FakeModel {
        fn provider(&self) -> &str {
            &self.provider
        }

        fn model_name(&self) -> &str {
            &self.model_name
        }

        async fn invoke(
            &self,
            _messages: &[BaseMessage],
            _request: &ModelRequestConfig,
        ) -> Result<ChatResponse> {
            Err(Error::unsupported(
                "the fake model does not call a provider",
            ))
        }

        fn stream(
            &self,
            _messages: Vec<BaseMessage>,
            _request: ModelRequestConfig,
        ) -> BoxStream<'static, Result<StreamChunk>> {
            Box::pin(futures::stream::empty())
        }
    }

    /// A provider that behaves like a built-in: it records the resolved config it
    /// was handed and builds a real (never-invoked) OpenAI-compatible client.
    struct RecordingProvider {
        name: &'static str,
        seen: Arc<Mutex<Vec<ModelClientConfig>>>,
        base_urls: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingProvider {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                seen: Arc::new(Mutex::new(Vec::new())),
                base_urls: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl ModelProvider for RecordingProvider {
        fn name(&self) -> &str {
            self.name
        }

        fn create(
            &self,
            client: &ModelClientConfig,
            request: &ModelRequestConfig,
        ) -> Result<Arc<dyn Model>> {
            let concrete = OpenAiCompatibleClient::new(client.clone(), request.clone())?;
            self.seen.lock().expect("not poisoned").push(client.clone());
            self.base_urls
                .lock()
                .expect("not poisoned")
                .push(concrete.base_url().to_owned());
            Ok(Arc::new(concrete))
        }
    }

    /// A provider that only records what it received.
    struct EchoProvider {
        name: &'static str,
        seen: Arc<Mutex<Vec<ModelClientConfig>>>,
    }

    impl ModelProvider for EchoProvider {
        fn name(&self) -> &str {
            self.name
        }

        fn create(
            &self,
            client: &ModelClientConfig,
            request: &ModelRequestConfig,
        ) -> Result<Arc<dyn Model>> {
            self.seen.lock().expect("not poisoned").push(client.clone());
            Ok(Arc::new(FakeModel {
                provider: client.provider.clone(),
                model_name: request.model_name.clone(),
            }))
        }
    }

    /// Serializes tests that mutate the process environment.
    fn env_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Clears the given variables for the guard's lifetime and restores them on
    /// drop.
    struct EnvGuard {
        _lock: MutexGuard<'static, ()>,
        saved: Vec<(&'static str, Option<String>)>,
    }

    impl EnvGuard {
        fn clearing(names: &[&'static str]) -> Self {
            let lock = env_lock();
            let saved: Vec<(&'static str, Option<String>)> = names
                .iter()
                .map(|name| (*name, std::env::var(name).ok()))
                .collect();
            for name in names {
                std::env::remove_var(name);
            }
            Self { _lock: lock, saved }
        }

        #[allow(clippy::unused_self)]
        fn set(&self, name: &str, value: &str) {
            std::env::set_var(name, value);
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in &self.saved {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }

    #[tokio::test]
    async fn init_model_dispatches_the_openai_flavor() {
        let client = ModelClientConfig::new("openai")
            .with_api_key("sk-test")
            .with_api_base("http://127.0.0.1:1234/v1");
        let request = ModelRequestConfig::new("gpt-4o-mini");

        let model = init_model(client, request)
            .await
            .expect("openai is built in");
        assert_eq!(model.provider(), "openai");
        assert_eq!(model.model_name(), "gpt-4o-mini");
    }

    #[tokio::test]
    async fn init_model_dispatches_the_anthropic_flavor() {
        let client = ModelClientConfig::new("anthropic")
            .with_api_key("sk-ant-test")
            .with_api_base("https://api.anthropic.com");
        let request = ModelRequestConfig::new("claude-sonnet-4");

        let model = init_model(client, request)
            .await
            .expect("anthropic is built in");
        assert_eq!(model.provider(), "anthropic");
        assert_eq!(model.model_name(), "claude-sonnet-4");
    }

    #[tokio::test]
    async fn init_model_accepts_vendor_aliases_and_separators() {
        let client = ModelClientConfig::new("DashScope")
            .with_api_key("sk-test")
            .with_api_base("https://dashscope.aliyuncs.com/compatible-mode/v1");
        let request = ModelRequestConfig::new("qwen-plus");

        let model = init_model(client, request)
            .await
            .expect("dashscope is qwen");
        assert_eq!(model.provider(), "qwen");
        assert_eq!(model.model_name(), "qwen-plus");
    }

    #[tokio::test]
    async fn init_model_reports_unknown_providers() {
        let registry_names = ModelProviderRegistry::global().names();
        let error = match init_model(
            ModelClientConfig::new("acme").with_api_key("sk-test"),
            ModelRequestConfig::new("acme-1"),
        )
        .await
        {
            Ok(_) => panic!("acme is not a known provider"),
            Err(error) => error,
        };

        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.to_string();
        assert!(message.contains("acme"), "message: {message}");
        for name in registry_names {
            assert!(
                message.contains(&name),
                "message must list {name}: {message}"
            );
        }
    }

    #[tokio::test]
    async fn init_model_resolves_configs_for_builtin_names_before_create() {
        let registry = ModelProviderRegistry::new();
        let provider = Arc::new(RecordingProvider::new("openai"));
        let seen = provider.seen.clone();
        let base_urls = provider.base_urls.clone();
        registry.register(provider);

        let client = ModelClientConfig::new("OpenAI")
            .with_api_key("sk-test")
            .with_api_base("http://127.0.0.1:1234/v1/");
        let model = init_model_in(&registry, client, ModelRequestConfig::new("gpt-4o-mini"))
            .await
            .expect("openai resolves");

        assert_eq!(model.provider(), "openai");
        assert_eq!(model.model_name(), "gpt-4o-mini");

        let recorded = seen.lock().expect("not poisoned").clone();
        let recorded = recorded.first().expect("create ran exactly once");
        assert_eq!(recorded.provider, "openai", "provider is normalized");
        assert_eq!(
            recorded.api_base.as_deref(),
            Some("http://127.0.0.1:1234/v1/"),
            "the configured api_base is passed through"
        );
        assert!(
            recorded.api_key.is_some(),
            "the configured credential is passed through"
        );
        assert_eq!(
            base_urls.lock().expect("not poisoned").as_slice(),
            ["http://127.0.0.1:1234/v1"],
            "the client strips the trailing slash from api_base"
        );
    }

    #[tokio::test]
    async fn init_model_passes_extension_configs_through_unresolved() {
        let registry = ModelProviderRegistry::new();
        let provider = Arc::new(EchoProvider {
            name: "acme",
            seen: Arc::new(Mutex::new(Vec::new())),
        });
        let seen = provider.seen.clone();
        registry.register(provider);

        // No key and no base: resolving this config through the static table would
        // fail with `Unsupported`, so success proves the factory left it alone.
        let model = init_model_in(
            &registry,
            ModelClientConfig::new("ACME"),
            ModelRequestConfig::new("acme-1"),
        )
        .await
        .expect("a registered extension provider is dispatched to");

        assert_eq!(model.provider(), "acme");
        assert_eq!(model.model_name(), "acme-1");

        let recorded = seen.lock().expect("not poisoned").clone();
        let recorded = recorded.first().expect("create ran exactly once");
        assert_eq!(recorded.provider, "acme", "only the name is normalized");
        assert!(
            recorded.api_base.is_none(),
            "no provider default or environment fallback is applied"
        );
        assert!(
            recorded.api_key.is_none(),
            "no environment credential is applied"
        );
    }

    #[test]
    fn init_model_with_uses_the_python_factory_defaults() {
        let client = factory_client_config("openai");

        assert_eq!(client.provider, "openai");
        assert_eq!(client.timeout, Some(Duration::from_secs(60)));
        assert!(!client.verify_ssl);
        assert_eq!(client.max_retries, 3);

        let strict = ModelClientConfig::default();
        assert_eq!(strict.timeout, Some(Duration::from_secs(360)));
        assert!(strict.verify_ssl);
        assert_eq!(strict.max_retries, 0);
    }

    #[tokio::test]
    async fn init_model_with_builds_a_client_for_the_named_provider() {
        let model = init_model_with("ollama", "llama3")
            .await
            .expect("ollama needs no credentials");

        assert_eq!(model.provider(), "ollama");
        assert_eq!(model.model_name(), "llama3");
    }

    #[tokio::test]
    async fn blank_provider_falls_back_to_the_environment_default() {
        let env = EnvGuard::clearing(&[env::MODEL_PROVIDER]);

        // Another test module may be rewriting the process environment at the same
        // time (its lock is module-private), so retry until one attempt runs
        // without interference.
        for _ in 0..20 {
            env.set(env::MODEL_PROVIDER, "ollama");
            let client = ModelClientConfig::default()
                .with_api_key("sk-unused")
                .with_api_base("http://127.0.0.1:11434/v1");
            if let Ok(model) = init_model(client, ModelRequestConfig::new("llama3")).await {
                if model.provider() == "ollama" {
                    assert_eq!(model.model_name(), "llama3");
                    return;
                }
            }
        }
        panic!("the environment default did not stay stable for 20 attempts");
    }

    #[test]
    fn requested_provider_prefers_the_config_then_the_environment() {
        assert_eq!(
            requested_provider(&ModelClientConfig::new("  DeepSeek ")),
            "DeepSeek",
            "the configured name is used verbatim and normalized later"
        );
        assert_eq!(
            requested_provider(&ModelClientConfig::default()),
            env::model_provider(),
            "a blank provider falls back to MODEL_PROVIDER"
        );
        assert_eq!(
            requested_provider(&ModelClientConfig::new("   ")),
            env::model_provider(),
            "a whitespace-only provider is blank"
        );
    }
}
