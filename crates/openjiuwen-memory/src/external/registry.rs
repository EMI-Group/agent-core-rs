//! Provider kinds, configuration, factories, and the [`ProviderRegistry`].
//!
//! One registry instance per application — injected, never a process-global — maps a
//! [`ProviderKind`] to the factory that builds it and a name to the live provider
//! instances. Registration is interior-mutable behind a `parking_lot::RwLock` that is
//! never held across an `.await` (every registry method is synchronous), so a
//! [`ProviderRegistry`] is `Send + Sync` and cheap to clone.
//!
//! The registry is deliberately tolerant:
//!
//! - An unknown [`ProviderKind`] deserializes to `ProviderKind::Custom` instead of
//!   failing, so a newer configuration file does not break an older build.
//! - Registering a factory or a provider twice **replaces** the previous entry (the
//!   `HashMap::insert` contract) and returns it; last registration wins.
//! - The first provider registered becomes the default; [`ProviderRegistry::select`]
//!   moves the default to the provider it just built.

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::{redact_json, REDACTED};
use parking_lot::RwLock;
use serde::de::Visitor;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::engine::MemoryEngine;

use super::local::{EngineProvider, LocalProviderFactory};
use super::provider::MemoryProvider;
use super::vendor::vendor_factories;

/// Option keys whose values are redacted in [`ProviderConfig`]'s `Debug` output.
const SENSITIVE_OPTION_KEYS: &[&str] = &[
    "api_key",
    "apikey",
    "access_key",
    "secret",
    "secret_key",
    "token",
    "password",
    "authorization",
];

/// The identity of a memory backend.
///
/// The serialized names are the Python provider ids: `agentarts`, `jiuwen_memory`,
/// `lakebase`, `mem0`, `openviking`, and `local` (Python's in-process
/// `OpenJiuwenMemoryProvider`). Common spelling variants (`agent_arts`, `lake_base`,
/// `open_viking`, `openjiuwen`, `engine`) are accepted on input. An id this crate does
/// not know becomes [`ProviderKind::Custom`] with the string preserved verbatim, so a
/// provider added by a newer release still round-trips through an older build.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProviderKind {
    /// Huawei `AgentArts` memory service.
    AgentArts,
    /// Jiuwen memory server or SDK.
    Jiuwen,
    /// LakeBase/DBay memory service.
    LakeBase,
    /// mem0 hosted memory.
    Mem0,
    /// `OpenViking` context database.
    OpenViking,
    /// The in-process engine adapter ([`EngineProvider`]).
    Local,
    /// Any provider id this crate does not know, kept verbatim.
    Custom(String),
}

impl ProviderKind {
    /// Every built-in kind, without the [`ProviderKind::Custom`] escape hatch.
    pub const BUILTIN: [Self; 6] = [
        Self::AgentArts,
        Self::Jiuwen,
        Self::LakeBase,
        Self::Mem0,
        Self::OpenViking,
        Self::Local,
    ];

    /// The Python provider id of this kind.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::AgentArts => "agentarts",
            Self::Jiuwen => "jiuwen_memory",
            Self::LakeBase => "lakebase",
            Self::Mem0 => "mem0",
            Self::OpenViking => "openviking",
            Self::Local => "local",
            Self::Custom(name) => name,
        }
    }

    /// Resolves a provider id, tolerating spelling variants and unknown names.
    ///
    /// Matching is ASCII-case-insensitive and ignores surrounding whitespace. An
    /// unrecognized name (blank included) becomes [`ProviderKind::Custom`] with the
    /// trimmed original string.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        let trimmed = name.trim();
        match trimmed.to_ascii_lowercase().as_str() {
            "agentarts" | "agent_arts" => Self::AgentArts,
            "jiuwen" | "jiuwen_memory" => Self::Jiuwen,
            "lakebase" | "lake_base" => Self::LakeBase,
            "mem0" => Self::Mem0,
            "openviking" | "open_viking" => Self::OpenViking,
            "local" | "engine" | "openjiuwen" => Self::Local,
            _ => Self::Custom(trimmed.to_owned()),
        }
    }
}

impl Default for ProviderKind {
    /// The in-process engine, matching the Python default backend.
    fn default() -> Self {
        Self::Local
    }
}

impl fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ProviderKind {
    type Err = std::convert::Infallible;

    /// Every string resolves; unknown names become [`ProviderKind::Custom`].
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self::from_name(s))
    }
}

impl Serialize for ProviderKind {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ProviderKind {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct KindVisitor;

        impl Visitor<'_> for KindVisitor {
            type Value = ProviderKind;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a memory provider id")
            }

            fn visit_str<E: serde::de::Error>(
                self,
                value: &str,
            ) -> std::result::Result<Self::Value, E> {
                Ok(ProviderKind::from_name(value))
            }
        }

        deserializer.deserialize_str(KindVisitor)
    }
}

/// How to reach one memory backend.
///
/// `name` is the registry key of the built provider; a blank `name` falls back to
/// [`ProviderKind::as_str`]. `api_key` is a live secret: [`fmt::Debug`] and
/// [`ProviderConfig::redacted`] mask it, while `Serialize` keeps it so a configuration
/// round-trips through JSON. `options` is the backend-specific remainder (tenant,
/// space, index name, …), which lets a vendor adapter be configured without changing
/// this struct.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ProviderConfig {
    /// Which backend to build.
    pub kind: ProviderKind,
    /// Registry name of the built provider; blank means [`ProviderKind::as_str`].
    pub name: String,
    /// Service endpoint or base URL, when the backend has one.
    pub endpoint: Option<String>,
    /// Credential handed to the backend; never printed by `Debug`.
    pub api_key: Option<String>,
    /// Per-request deadline serialized as whole milliseconds.
    #[serde(with = "timeout_millis", skip_serializing_if = "Option::is_none")]
    pub timeout: Option<Duration>,
    /// Backend-specific settings the typed fields do not cover.
    pub options: serde_json::Map<String, Value>,
}

impl Default for ProviderConfig {
    /// A local configuration with no credential and no backend-specific options.
    fn default() -> Self {
        Self {
            kind: ProviderKind::default(),
            name: String::new(),
            endpoint: None,
            api_key: None,
            timeout: None,
            options: serde_json::Map::new(),
        }
    }
}

impl fmt::Debug for ProviderConfig {
    /// Renders the configuration with the credential and any sensitive option value
    /// replaced by [`REDACTED`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("kind", &self.kind)
            .field("name", &self.name)
            .field("endpoint", &self.endpoint)
            .field("api_key", &self.api_key.as_ref().map(|_| REDACTED))
            .field("timeout", &self.timeout)
            .field("options", &self.redacted_options())
            .finish()
    }
}

impl ProviderConfig {
    /// A configuration for `kind` with every optional field empty.
    #[must_use]
    pub fn new(kind: ProviderKind) -> Self {
        Self {
            kind,
            ..Self::default()
        }
    }

    /// The registry name the built provider takes.
    #[must_use]
    pub fn provider_name(&self) -> &str {
        if self.name.trim().is_empty() {
            self.kind.as_str()
        } else {
            &self.name
        }
    }

    /// Returns the configuration with `name` set.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Returns the configuration with `endpoint` set.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Returns the configuration with `api_key` set.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Returns the configuration with a per-request deadline.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Returns the configuration with one backend-specific option inserted.
    #[must_use]
    pub fn with_option(mut self, key: impl Into<String>, value: Value) -> Self {
        self.options.insert(key.into(), value);
        self
    }

    /// Reads one backend-specific option.
    #[must_use]
    pub fn option(&self, key: &str) -> Option<&Value> {
        self.options.get(key)
    }

    /// Returns a copy whose `api_key` is replaced by [`REDACTED`].
    ///
    /// The copy is safe to log or attach to a diagnostic event. It is deliberately not
    /// meant for building a provider: the credential is destroyed. `options` are left
    /// untouched because a backend may need them; use `Debug` when the whole output is
    /// going to a log sink.
    #[must_use]
    pub fn redacted(&self) -> Self {
        let mut copy = self.clone();
        if copy.api_key.is_some() {
            copy.api_key = Some(REDACTED.to_owned());
        }
        copy
    }

    /// The option map with every sensitive value replaced by [`REDACTED`].
    fn redacted_options(&self) -> Value {
        redact_json(
            &Value::Object(self.options.clone()),
            Some(SENSITIVE_OPTION_KEYS),
        )
    }
}

/// Serializes `Option<Duration>` as whole milliseconds instead of `{secs, nanos}`.
mod timeout_millis {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    /// Writes the deadline as `Some(millis)` / `None`.
    // Serde's `with` contract passes `&Option<T>`; clippy's `ref_option` preference does
    // not apply to a serializer shim.
    #[allow(clippy::ref_option)]
    pub(super) fn serialize<S: Serializer>(
        value: &Option<Duration>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(duration) => {
                let millis = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
                serializer.serialize_some(&millis)
            }
            None => serializer.serialize_none(),
        }
    }

    /// Reads a millisecond count back into a [`Duration`].
    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Duration>, D::Error> {
        let millis = Option::<u64>::deserialize(deserializer)?;
        Ok(millis.map(Duration::from_millis))
    }
}

/// Builds one [`MemoryProvider`] from a [`ProviderConfig`].
///
/// A factory is synchronous and side-effect free: it validates the configuration and
/// constructs the handle, but performs no network I/O. A backend that needs something the
/// configuration does not carry still builds a handle whose operations return
/// [`openjiuwen_common::error::Error::Unsupported`] — a REST vendor with no `endpoint`, or
/// the SDK-only `agentarts` adapter — mirroring Python's lazy client acquisition; a
/// configuration that cannot be interpreted at all returns
/// [`openjiuwen_common::error::Error::Validation`].
pub trait ProviderFactory: Send + Sync {
    /// The kind this factory builds.
    fn kind(&self) -> ProviderKind;

    /// Builds a provider from `config`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] when `config` does not
    /// describe a backend this factory can produce. A backend the configuration leaves
    /// incomplete still builds: the resulting handle reports
    /// [`openjiuwen_common::error::Error::Unsupported`] from each operation instead.
    fn build(&self, config: &ProviderConfig) -> Result<Arc<dyn MemoryProvider>>;
}

/// The shared mutable state of a [`ProviderRegistry`].
#[derive(Default)]
struct RegistryState {
    /// One factory per kind (a later registration replaces an earlier one).
    factories: HashMap<ProviderKind, Arc<dyn ProviderFactory>>,
    /// Live provider instances keyed by registry name.
    providers: HashMap<String, Arc<dyn MemoryProvider>>,
    /// The name [`ProviderRegistry::default_provider`] returns.
    default_name: Option<String>,
}

/// The application-scoped map of provider kinds, factories, and live instances.
///
/// The registry owns no global state: build one, inject it as an `Arc`, and register
/// factories or ready-made providers. Clones share the same state.
///
/// ```
/// use openjiuwen_memory::external::{ProviderKind, ProviderRegistry};
///
/// let registry = ProviderRegistry::with_vendor_stubs();
/// let provider = registry
///     .build(&openjiuwen_memory::external::ProviderConfig::new(ProviderKind::Mem0))
///     .expect("the mem0 factory is registered");
/// assert_eq!(provider.name(), "mem0");
/// ```
#[derive(Clone, Default)]
pub struct ProviderRegistry {
    /// Factories, instances, and the default name.
    state: Arc<RwLock<RegistryState>>,
}

impl ProviderRegistry {
    /// Builds an empty registry: no factory and no provider.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a registry with the five vendor factories registered.
    ///
    /// Every adapter builds successfully, so a deployment can be configured for a vendor
    /// and fail loudly at first use instead of at registration. Four of the five (`mem0`,
    /// `jiuwen_memory`, `lakebase`, `openviking`) perform real requests once an `endpoint`
    /// is configured; without one the handle is inert and every operation returns
    /// [`openjiuwen_common::error::Error::Unsupported`] stating that no endpoint is
    /// configured. `agentarts` is SDK-only, so it reports `Unsupported` naming
    /// `agentarts.sdk.memory.MemoryClient` for every operation whatever the configuration.
    #[must_use]
    pub fn with_vendor_stubs() -> Self {
        let registry = Self::new();
        for factory in vendor_factories() {
            registry.register_factory(factory);
        }
        registry
    }

    /// Builds a registry with the vendor factories plus the local engine provider.
    ///
    /// The `local` provider is registered as a ready instance and becomes the default;
    /// [`ProviderRegistry::build`] with [`ProviderKind::Local`] returns further providers
    /// over the same engine.
    #[must_use]
    pub fn with_local(engine: Arc<MemoryEngine>) -> Self {
        let registry = Self::with_vendor_stubs();
        registry.register_factory(Arc::new(LocalProviderFactory::new(Arc::clone(&engine))));
        registry.register("local", Arc::new(EngineProvider::new(engine)));
        registry
    }

    /// Registers `factory` for its kind, returning the factory it replaced.
    ///
    /// A duplicate kind replaces the previous factory (`HashMap::insert` semantics).
    pub fn register_factory(
        &self,
        factory: Arc<dyn ProviderFactory>,
    ) -> Option<Arc<dyn ProviderFactory>> {
        self.state.write().factories.insert(factory.kind(), factory)
    }

    /// Registers `provider` under `name`, returning the instance it replaced.
    ///
    /// A duplicate name replaces the previous instance. The first registration also
    /// becomes the default provider; later registrations leave the default alone.
    pub fn register(
        &self,
        name: impl Into<String>,
        provider: Arc<dyn MemoryProvider>,
    ) -> Option<Arc<dyn MemoryProvider>> {
        let name = name.into();
        let mut state = self.state.write();
        let previous = state.providers.insert(name.clone(), provider);
        if state.default_name.is_none() {
            state.default_name = Some(name);
        }
        previous
    }

    /// Returns the provider registered under `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn MemoryProvider>> {
        self.state.read().providers.get(name).cloned()
    }

    /// Returns every registered provider name, sorted.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.state.read().providers.keys().cloned().collect();
        names.sort();
        names
    }

    /// Returns the default provider, if one is registered.
    #[must_use]
    pub fn default_provider(&self) -> Option<Arc<dyn MemoryProvider>> {
        let state = self.state.read();
        let name = state.default_name.as_ref()?;
        state.providers.get(name).cloned()
    }

    /// Makes the registered provider `name` the default.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::NotFound`] when `name` is not
    /// registered.
    pub fn set_default(&self, name: &str) -> Result<()> {
        let mut state = self.state.write();
        if !state.providers.contains_key(name) {
            return Err(Error::not_found(format!(
                "memory provider `{name}` is not registered"
            )));
        }
        state.default_name = Some(name.to_owned());
        Ok(())
    }

    /// Builds a provider from `config` using the factory registered for its kind.
    ///
    /// Building does not register the provider and does not change the default; use
    /// [`ProviderRegistry::select`] for that.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Unsupported`] when no factory is
    /// registered for `config.kind`, and the factory's own error otherwise.
    pub fn build(&self, config: &ProviderConfig) -> Result<Arc<dyn MemoryProvider>> {
        let factory = { self.state.read().factories.get(&config.kind).cloned() };
        let Some(factory) = factory else {
            return Err(Error::unsupported(format!(
                "no memory-provider factory is registered for kind `{}`",
                config.kind
            )));
        };
        factory.build(config)
    }

    /// Builds the provider `config` describes, registers it under
    /// [`ProviderConfig::provider_name`], makes it the default, and returns it.
    ///
    /// # Errors
    ///
    /// See [`ProviderRegistry::build`].
    pub fn select(&self, config: &ProviderConfig) -> Result<Arc<dyn MemoryProvider>> {
        let provider = self.build(config)?;
        self.register(config.provider_name(), Arc::clone(&provider));
        Ok(provider)
    }
}

impl fmt::Debug for ProviderRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.read();
        let mut kinds: Vec<&str> = state.factories.keys().map(ProviderKind::as_str).collect();
        kinds.sort_unstable();
        let mut names: Vec<&str> = state.providers.keys().map(String::as_str).collect();
        names.sort_unstable();
        f.debug_struct("ProviderRegistry")
            .field("factories", &kinds)
            .field("providers", &names)
            .field("default", &state.default_name)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use openjiuwen_common::error::Error as CommonError;

    use crate::config::MemoryEngineConfig;
    use crate::store::MemoryStores;

    fn engine() -> Arc<MemoryEngine> {
        MemoryEngine::new(MemoryEngineConfig::default(), MemoryStores::in_memory()).expect("engine")
    }

    fn provider(name: &str) -> Arc<dyn MemoryProvider> {
        Arc::new(EngineProvider::with_name(engine(), name))
    }

    #[test]
    fn known_kinds_use_the_python_provider_ids() {
        assert_eq!(ProviderKind::AgentArts.as_str(), "agentarts");
        assert_eq!(ProviderKind::Jiuwen.as_str(), "jiuwen_memory");
        assert_eq!(ProviderKind::LakeBase.as_str(), "lakebase");
        assert_eq!(ProviderKind::Mem0.as_str(), "mem0");
        assert_eq!(ProviderKind::OpenViking.as_str(), "openviking");
        assert_eq!(ProviderKind::Local.as_str(), "local");
        assert_eq!(ProviderKind::Custom("acme".to_owned()).as_str(), "acme");
        assert_eq!(ProviderKind::default(), ProviderKind::Local);
        assert_eq!(ProviderKind::BUILTIN.len(), 6);
    }

    #[test]
    fn from_name_tolerates_variants_and_preserves_unknowns() {
        for (name, expected) in [
            ("agentarts", ProviderKind::AgentArts),
            ("AgentArts", ProviderKind::AgentArts),
            (" agent_arts ", ProviderKind::AgentArts),
            ("jiuwen", ProviderKind::Jiuwen),
            ("jiuwen_memory", ProviderKind::Jiuwen),
            ("lakebase", ProviderKind::LakeBase),
            ("lake_base", ProviderKind::LakeBase),
            ("mem0", ProviderKind::Mem0),
            ("openviking", ProviderKind::OpenViking),
            ("open_viking", ProviderKind::OpenViking),
            ("local", ProviderKind::Local),
            ("engine", ProviderKind::Local),
            ("openjiuwen", ProviderKind::Local),
        ] {
            assert_eq!(ProviderKind::from_name(name), expected, "{name}");
        }
        assert_eq!(
            ProviderKind::from_name("Acme Memory"),
            ProviderKind::Custom("Acme Memory".to_owned())
        );
        assert_eq!(
            ProviderKind::from_name("  "),
            ProviderKind::Custom(String::new())
        );
        assert_eq!(
            "acme".parse::<ProviderKind>().expect("infallible"),
            ProviderKind::Custom("acme".to_owned())
        );
    }

    #[test]
    fn kind_serde_round_trips_every_builtin_and_unknown_names() {
        for kind in ProviderKind::BUILTIN {
            let json = serde_json::to_value(&kind).expect("serialize");
            assert_eq!(json, serde_json::json!(kind.as_str()));
            let restored: ProviderKind = serde_json::from_value(json).expect("deserialize");
            assert_eq!(restored, kind);
        }
        let custom = ProviderKind::Custom("acme".to_owned());
        assert_eq!(
            serde_json::to_value(&custom).expect("serialize"),
            serde_json::json!("acme")
        );
        let restored: ProviderKind =
            serde_json::from_str("\"unknown_vendor\"").expect("deserialize");
        assert_eq!(restored, ProviderKind::Custom("unknown_vendor".to_owned()));
        assert_eq!(format!("{}", ProviderKind::Jiuwen), "jiuwen_memory");
    }

    #[test]
    fn config_defaults_and_builders() {
        let config = ProviderConfig::default();
        assert_eq!(config.kind, ProviderKind::Local);
        assert_eq!(config.provider_name(), "local");
        assert!(config.endpoint.is_none());
        assert!(config.api_key.is_none());
        assert!(config.timeout.is_none());
        assert!(config.options.is_empty());

        let config = ProviderConfig::new(ProviderKind::Mem0)
            .with_name("team-mem0")
            .with_endpoint("https://api.mem0.ai")
            .with_api_key("sk-live")
            .with_timeout(Duration::from_millis(2500))
            .with_option("user_id", serde_json::json!("alice"));
        assert_eq!(config.provider_name(), "team-mem0");
        assert_eq!(config.option("user_id"), Some(&serde_json::json!("alice")));
        assert_eq!(config.option("absent"), None);

        let blank = ProviderConfig::new(ProviderKind::Mem0).with_name("   ");
        assert_eq!(blank.provider_name(), "mem0");
    }

    #[test]
    fn config_serde_round_trips_and_serializes_timeout_as_millis() {
        let config = ProviderConfig::new(ProviderKind::LakeBase)
            .with_endpoint("http://localhost:8080/api/v1")
            .with_api_key("secret")
            .with_timeout(Duration::from_millis(2500))
            .with_option("base_id", serde_json::json!("mem_default"));
        let json = serde_json::to_value(&config).expect("serialize");
        assert_eq!(json["kind"], serde_json::json!("lakebase"));
        assert_eq!(json["timeout"], serde_json::json!(2500));
        assert_eq!(json["api_key"], serde_json::json!("secret"));
        let restored: ProviderConfig = serde_json::from_value(json).expect("deserialize");
        assert_eq!(restored, config);

        let restored: ProviderConfig =
            serde_json::from_value(serde_json::json!({})).expect("defaults");
        assert_eq!(restored, ProviderConfig::default());
        let restored: ProviderConfig = serde_json::from_str(
            r#"{"kind": "acme", "name": "", "timeout": 10, "tenant": "ignored"}"#,
        )
        .expect("unknown kind is tolerated");
        assert_eq!(restored.kind, ProviderKind::Custom("acme".to_owned()));
        assert_eq!(restored.timeout, Some(Duration::from_millis(10)));
    }

    #[test]
    fn config_debug_and_redacted_hide_the_credential() {
        let config = ProviderConfig::new(ProviderKind::Mem0)
            .with_api_key("sk-live-secret")
            .with_option("token", serde_json::json!("opt-secret"))
            .with_option("space", serde_json::json!("workspace"));

        let debug = format!("{config:?}");
        assert!(debug.contains("ProviderConfig"));
        assert!(!debug.contains("sk-live-secret"), "{debug}");
        assert!(!debug.contains("opt-secret"), "{debug}");
        assert!(debug.contains(REDACTED), "{debug}");
        assert!(debug.contains("workspace"), "{debug}");

        let redacted = config.redacted();
        assert_eq!(redacted.api_key.as_deref(), Some(REDACTED));
        assert_eq!(redacted.endpoint, config.endpoint);
        assert_eq!(redacted.options, config.options);
        let no_key = ProviderConfig::new(ProviderKind::Local);
        assert!(no_key.redacted().api_key.is_none());
    }

    struct TestFactory {
        kind: ProviderKind,
        name: &'static str,
        engine: Arc<MemoryEngine>,
    }

    impl TestFactory {
        fn new(kind: ProviderKind, name: &'static str, engine: Arc<MemoryEngine>) -> Self {
            Self { kind, name, engine }
        }
    }

    impl ProviderFactory for TestFactory {
        fn kind(&self) -> ProviderKind {
            self.kind.clone()
        }

        fn build(&self, _config: &ProviderConfig) -> Result<Arc<dyn MemoryProvider>> {
            Ok(Arc::new(EngineProvider::with_name(
                Arc::clone(&self.engine),
                self.name,
            )))
        }
    }

    #[test]
    fn an_empty_registry_has_no_provider_and_builds_nothing() {
        let registry = ProviderRegistry::new();
        assert!(registry.names().is_empty(), "{:?}", registry.names());
        assert!(registry.default_provider().is_none());
        assert!(registry.get("local").is_none());
        assert!(format!("{registry:?}").contains("ProviderRegistry"));

        let error = registry
            .build(&ProviderConfig::new(ProviderKind::Mem0))
            .err()
            .expect("no factory");
        assert!(matches!(error, CommonError::Unsupported(_)), "{error:?}");
        assert!(error.to_string().contains("mem0"), "{error}");
    }

    #[test]
    fn the_first_registered_provider_is_the_default_and_duplicates_replace() {
        let registry = ProviderRegistry::new();
        assert!(registry.register("first", provider("first")).is_none());
        assert_eq!(registry.names(), vec!["first"]);
        assert_eq!(
            registry.default_provider().expect("default").name(),
            "first"
        );

        assert!(registry.register("second", provider("second")).is_none());
        assert_eq!(registry.names(), vec!["first", "second"]);
        assert_eq!(
            registry.default_provider().expect("default").name(),
            "first"
        );

        let previous = registry.register("first", provider("replacement"));
        assert!(previous.is_some());
        assert_eq!(registry.get("first").expect("first").name(), "replacement");
        assert_eq!(
            registry.default_provider().expect("default").name(),
            "replacement"
        );
        assert_eq!(registry.names(), vec!["first", "second"]);
    }

    #[test]
    fn set_default_requires_a_registered_provider() {
        let registry = ProviderRegistry::new();
        registry.register("first", provider("first"));
        registry.register("second", provider("second"));
        registry.set_default("second").expect("registered");
        assert_eq!(
            registry.default_provider().expect("default").name(),
            "second"
        );
        let error = registry.set_default("absent").expect_err("not registered");
        assert!(matches!(error, CommonError::NotFound(_)), "{error:?}");
    }

    #[test]
    fn duplicate_factory_registration_replaces_and_the_last_one_builds() {
        let registry = ProviderRegistry::new();
        let kind = ProviderKind::Custom("acme".to_owned());
        assert!(registry
            .register_factory(Arc::new(TestFactory::new(kind.clone(), "v1", engine())))
            .is_none());
        let built = registry
            .build(&ProviderConfig::new(kind.clone()))
            .expect("build");
        assert_eq!(built.name(), "v1");

        let previous =
            registry.register_factory(Arc::new(TestFactory::new(kind.clone(), "v2", engine())));
        assert!(previous.is_some());
        let built = registry.build(&ProviderConfig::new(kind)).expect("build");
        assert_eq!(built.name(), "v2");
    }

    #[test]
    fn build_is_pure_and_select_registers_and_sets_the_default() {
        let registry = ProviderRegistry::new();
        registry.register_factory(Arc::new(TestFactory::new(
            ProviderKind::Mem0,
            "built",
            engine(),
        )));

        let config = ProviderConfig::new(ProviderKind::Mem0);
        let built = registry.build(&config).expect("build");
        assert_eq!(built.name(), "built");
        assert!(registry.names().is_empty(), "{:?}", registry.names());
        assert!(registry.default_provider().is_none());

        let selected = registry.select(&config).expect("select");
        assert_eq!(selected.name(), "built");
        assert_eq!(registry.names(), vec!["mem0"]);
        assert_eq!(
            registry.default_provider().expect("default").name(),
            "built"
        );

        let named = ProviderConfig::new(ProviderKind::Mem0).with_name("secondary");
        let selected = registry.select(&named).expect("select named");
        assert_eq!(selected.name(), "built");
        assert_eq!(registry.names(), vec!["mem0", "secondary"]);
        assert_eq!(
            registry.default_provider().expect("default").name(),
            "built"
        );
        assert!(registry.get("secondary").is_some());
    }

    #[test]
    fn with_vendor_stubs_registers_the_five_vendor_factories() {
        let registry = ProviderRegistry::with_vendor_stubs();
        assert!(registry.names().is_empty(), "{:?}", registry.names());
        for kind in [
            ProviderKind::AgentArts,
            ProviderKind::Jiuwen,
            ProviderKind::LakeBase,
            ProviderKind::Mem0,
            ProviderKind::OpenViking,
        ] {
            let built = registry
                .build(&ProviderConfig::new(kind.clone()))
                .unwrap_or_else(|error| panic!("{kind} must build: {error}"));
            assert_eq!(built.name(), kind.as_str());
        }
        // The local kind has no factory until an engine is supplied.
        let error = registry
            .build(&ProviderConfig::new(ProviderKind::Local))
            .err()
            .expect("no local factory");
        assert!(matches!(error, CommonError::Unsupported(_)), "{error:?}");
    }

    #[tokio::test]
    async fn with_local_registers_the_default_and_shares_the_engine() {
        let engine = engine();
        let registry = ProviderRegistry::with_local(Arc::clone(&engine));
        assert_eq!(registry.names(), vec!["local"]);
        assert_eq!(
            registry.default_provider().expect("default").name(),
            "local"
        );

        let local = registry
            .build(&ProviderConfig::new(ProviderKind::Local))
            .expect("local factory");
        assert_eq!(local.name(), "local");
        assert!(local.capabilities().keyword_search);

        let scope = crate::record::MemoryScope::new("alice", "ws");
        local
            .add(
                &scope,
                vec![crate::record::NewMemory::semantic("shared engine")],
                &crate::record::AddOptions::default(),
            )
            .await
            .expect("add");
        assert_eq!(
            engine
                .list(&crate::record::MemoryFilter::default())
                .await
                .expect("list")
                .len(),
            1
        );

        let named = registry
            .build(&ProviderConfig::new(ProviderKind::Local).with_name("secondary"))
            .expect("named local");
        assert_eq!(named.name(), "secondary");

        // Duplicate registration on select replaces and re-defaults.
        let selected = registry
            .select(&ProviderConfig::new(ProviderKind::Local))
            .expect("select local");
        assert_eq!(selected.name(), "local");
    }

    #[test]
    fn the_registry_is_send_sync_and_clone_shares_state() {
        fn assert_send_sync_static<T: Send + Sync + 'static>() {}
        assert_send_sync_static::<ProviderRegistry>();

        let registry = ProviderRegistry::new();
        registry.register("first", provider("first"));
        let clone = registry.clone();
        assert_eq!(clone.names(), vec!["first"]);
        clone.register("second", provider("second"));
        assert_eq!(registry.names(), vec!["first", "second"]);
    }
}
