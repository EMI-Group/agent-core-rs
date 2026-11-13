//! The provider interface the registry stores and the factory resolves.
//!
//! Python keys providers by name and probes them at runtime: `isinstance(provider,
//! HarnessProvider)` guards `card`/`create`, and `isinstance(provider, ModelConfigProvider)`
//! decides whether the optional model-configuration hook is available. Rust cannot specialize on
//! an optional trait implementation, so a single [`ProviderSpec`] interface is what gets
//! registered: it always exposes `card`/`create` and advertises the optional hook through
//! [`ProviderSpec::configures_models`]. The two wrapper constructors — [`plain_provider`] and
//! [`model_configuring_provider`] — adapt any [`HarnessProvider`], which is the Rust equivalent of
//! Python's duck typing (documented deviation).
//!
//! There is deliberately no blanket `impl<T: HarnessProvider> ProviderSpec for T`: it would
//! collide with the model-configuring specialization, and a provider that *does* configure models
//! would silently lose the hook. Registration is explicit so the chosen behavior is visible at the
//! call site.
//!
//! [`ProviderDescriptor`] is the introspection view of one registry entry: registry name, card
//! name, implementation version, the built-in / implemented flags, whether it configures models,
//! and the external requirement of a deferred provider.
use std::sync::Arc;

use openjiuwen_harness_protocol::{
    HarnessCard, HarnessError, HarnessProtocol, HarnessProvider, HarnessResult, JsonObject,
    ModelChoice, ModelConfigProvider,
};
use serde::Serialize;

/// Registry names of the harness providers this crate declares (Python `PROVIDER_NAMES`).
///
/// These are lookup keys, not harness identity: the card name a provider reports is separate
/// metadata (`native` serves the card `openjiuwen-native`, `claudecode` the card `claude-code`).
pub mod provider_names {
    /// In-process native openJiuwen harness (`openjiuwen-harness`).
    pub const NATIVE: &str = "native";
    /// Native v2 harness hosted by `openjiuwen-agent-teams`; needs the `native-v2` feature.
    pub const NATIVE_V2: &str = "native_v2";
    /// Claude Code harness; deferred vendor stub.
    pub const CLAUDE_CODE: &str = "claudecode";
    /// Codex harness; deferred vendor stub.
    pub const CODEX: &str = "codex";
    /// `DeepSeek` harness; deferred vendor stub.
    pub const DSH: &str = "dsh";
}

/// Every declared registry name, in Python `PROVIDER_NAMES` order.
const BUILTIN_PROVIDER_NAMES: [&str; 5] = [
    provider_names::NATIVE,
    provider_names::NATIVE_V2,
    provider_names::CLAUDE_CODE,
    provider_names::CODEX,
    provider_names::DSH,
];

/// Returns whether `name` is one of the provider names this crate declares.
///
/// Declared is not the same as usable: a name whose feature is compiled out (`native-v2` for the
/// agent-teams provider, `vendor` for the three vendor stubs) is known but cannot be resolved, so
/// resolution reports the missing feature.
#[must_use]
pub fn is_builtin_provider(name: &str) -> bool {
    BUILTIN_PROVIDER_NAMES.contains(&name)
}

/// The registered provider interface: harness identity, construction, and the optional model hook.
///
/// A provider only *constructs*: [`ProviderSpec::create`] returns an unstarted harness, and
/// [`HarnessProtocol::start`] is a separate call.
pub trait ProviderSpec: Send + Sync {
    /// Returns metadata for the harnesses created by this provider.
    #[must_use]
    fn card(&self) -> &HarnessCard;

    /// Validates `config` and creates an unstarted harness.
    ///
    /// `config` is the provider-defined create mapping; the provider owns its schema and must
    /// reject values it cannot run.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Validation`] when `config` is not a valid configuration for this
    /// provider, and [`HarnessError::Protocol`] when construction fails.
    // The harness SPI hands the create mapping over by value (DESIGN-fixed signature), so the
    // owned parameter is forwarded unchanged instead of being taken by reference.
    #[allow(clippy::needless_pass_by_value)]
    fn create(&self, config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>>;

    /// Returns whether this provider implements the model-configuration hook.
    ///
    /// The default is `false`: the host must then create the harness from configuration alone and
    /// must not call [`ProviderSpec::model_config`].
    #[must_use]
    fn configures_models(&self) -> bool {
        false
    }

    /// Returns the create fragment that runs `choice` (optionally with a `fallback` model).
    ///
    /// # Errors
    ///
    /// The default implementation reports [`HarnessError::UnsupportedCapability`], because a
    /// provider without the hook cannot map a model choice onto its configuration.
    fn model_config(
        &self,
        _choice: &ModelChoice,
        _fallback: Option<&ModelChoice>,
    ) -> HarnessResult<JsonObject> {
        Err(HarnessError::unsupported_capability(
            "harness provider does not configure models",
        ))
    }
}

/// Plain adapter: forwards to a [`HarnessProvider`] and keeps the default model-configuration
/// behavior.
#[derive(Debug)]
struct PlainProvider<P> {
    /// The adapted provider.
    provider: P,
}

impl<P: HarnessProvider> ProviderSpec for PlainProvider<P> {
    fn card(&self) -> &HarnessCard {
        self.provider.card()
    }

    fn create(&self, config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>> {
        self.provider.create(config)
    }
}

/// Model-configuring adapter: forwards to a provider that implements both SPI traits.
#[derive(Debug)]
struct ModelConfiguringProvider<P> {
    /// The adapted provider.
    provider: P,
}

impl<P: HarnessProvider + ModelConfigProvider> ProviderSpec for ModelConfiguringProvider<P> {
    fn card(&self) -> &HarnessCard {
        self.provider.card()
    }

    fn create(&self, config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>> {
        self.provider.create(config)
    }

    fn configures_models(&self) -> bool {
        true
    }

    fn model_config(
        &self,
        choice: &ModelChoice,
        fallback: Option<&ModelChoice>,
    ) -> HarnessResult<JsonObject> {
        self.provider.model_config(choice, fallback)
    }
}

/// Wraps `provider` as a registered [`ProviderSpec`] without a model-configuration hook.
///
/// Use this for providers that only construct harnesses from configuration (Python providers that
/// are not `ModelConfigProvider` instances).
#[must_use]
pub fn plain_provider<P: HarnessProvider + 'static>(provider: P) -> Arc<dyn ProviderSpec> {
    Arc::new(PlainProvider { provider })
}

/// Wraps `provider` as a registered [`ProviderSpec`] that also configures models.
///
/// The returned spec reports `configures_models() == true` and forwards
/// [`ProviderSpec::model_config`] to the provider's [`ModelConfigProvider`] implementation.
#[must_use]
pub fn model_configuring_provider<P: HarnessProvider + ModelConfigProvider + 'static>(
    provider: P,
) -> Arc<dyn ProviderSpec> {
    Arc::new(ModelConfiguringProvider { provider })
}

/// Introspection view of one registry entry.
///
/// Serialized `snake_case` so a CLI can print the registry as JSON; `requires` names the external
/// prerequisite of a deferred provider (an SDK, a CLI, or a sibling crate) and is [`None`] when the
/// provider is implemented in this build.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProviderDescriptor {
    /// Registry name (the lookup key, e.g. `native`).
    pub name: String,
    /// Harness identity the provider reports in its card (e.g. `openjiuwen-native`).
    pub card_name: String,
    /// Implementation version reported by the card.
    pub implementation_version: String,
    /// Whether the name is declared by this crate.
    pub builtin: bool,
    /// Whether this build can construct harnesses through the provider.
    pub implemented: bool,
    /// Whether the provider implements the model-configuration hook.
    pub configures_models: bool,
    /// External prerequisite of a deferred provider, if any.
    pub requires: Option<String>,
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use openjiuwen_harness_protocol::{
        HarnessCard, HarnessError, HarnessErrorKind, HarnessProtocol, HarnessProvider,
        HarnessResult, JsonObject, JsonValue, ModelChoice, ModelConfigProvider,
    };

    use super::{
        is_builtin_provider, model_configuring_provider, plain_provider, provider_names,
        ProviderDescriptor,
    };

    /// Card both fakes report.
    fn fake_card() -> HarnessCard {
        HarnessCard::new("fake-harness", "0.0.1")
    }

    /// Provider without a model-configuration hook.
    #[derive(Debug)]
    struct FakeProvider {
        /// Card reported by [`HarnessProvider::card`].
        card: HarnessCard,
    }

    impl FakeProvider {
        fn new() -> Self {
            Self { card: fake_card() }
        }
    }

    impl HarnessProvider for FakeProvider {
        fn card(&self) -> &HarnessCard {
            &self.card
        }

        fn create(&self, _config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>> {
            Err(HarnessError::validation("fake harness cannot be created"))
        }
    }

    /// Provider that also implements the model-configuration hook.
    #[derive(Debug)]
    struct ModelledFakeProvider {
        /// Card reported by [`HarnessProvider::card`].
        card: HarnessCard,
    }

    impl HarnessProvider for ModelledFakeProvider {
        fn card(&self) -> &HarnessCard {
            &self.card
        }

        fn create(&self, _config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>> {
            Err(HarnessError::validation("fake harness cannot be created"))
        }
    }

    impl ModelConfigProvider for ModelledFakeProvider {
        fn model_config(
            &self,
            choice: &ModelChoice,
            fallback: Option<&ModelChoice>,
        ) -> HarnessResult<JsonObject> {
            let mut fragment = JsonObject::new();
            if let Some(model) = &choice.model {
                fragment.insert("model".to_owned(), JsonValue::String(model.clone()));
            }
            if let Some(fallback) = fallback {
                fragment.insert(
                    "fallback".to_owned(),
                    JsonValue::String(fallback.model.clone().unwrap_or_default()),
                );
            }
            Ok(fragment)
        }
    }

    #[test]
    fn plain_provider_reports_no_model_configuration() {
        let spec = plain_provider(FakeProvider::new());
        assert!(!spec.configures_models());
        assert_eq!(spec.card().name, "fake-harness");
    }

    #[test]
    fn plain_provider_keeps_the_default_model_config_error() {
        let spec = plain_provider(FakeProvider::new());
        let choice = ModelChoice {
            model: Some("gpt-5".to_owned()),
            effort: None,
            endpoint: None,
        };
        let error = spec
            .model_config(&choice, None)
            .expect_err("a plain provider must not configure models");
        assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
        assert_eq!(
            error.message(),
            "harness provider does not configure models"
        );
    }

    #[test]
    fn plain_provider_forwards_create() {
        let spec = plain_provider(FakeProvider::new());
        let error = spec
            .create(JsonObject::new())
            .err()
            .expect("the fake provider cannot create a harness");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(error.message(), "fake harness cannot be created");
    }

    #[test]
    fn model_configuring_provider_reports_and_forwards_the_hook() {
        let spec = model_configuring_provider(ModelledFakeProvider { card: fake_card() });
        assert!(spec.configures_models());
        assert_eq!(spec.card().name, "fake-harness");

        let choice = ModelChoice {
            model: Some("gpt-5".to_owned()),
            effort: None,
            endpoint: None,
        };
        let fallback = ModelChoice {
            model: Some("gpt-5-mini".to_owned()),
            effort: None,
            endpoint: None,
        };
        let fragment = spec
            .model_config(&choice, Some(&fallback))
            .expect("the modelled provider returns a fragment");
        assert_eq!(
            fragment.get("model"),
            Some(&JsonValue::String("gpt-5".to_owned()))
        );
        assert_eq!(
            fragment.get("fallback"),
            Some(&JsonValue::String("gpt-5-mini".to_owned()))
        );
    }

    #[test]
    fn model_configuring_provider_forwards_create() {
        let spec = model_configuring_provider(ModelledFakeProvider { card: fake_card() });
        let error = spec
            .create(JsonObject::new())
            .err()
            .expect("the fake provider cannot create a harness");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
    }

    #[test]
    fn builtin_names_are_recognized() {
        for name in [
            provider_names::NATIVE,
            provider_names::NATIVE_V2,
            provider_names::CLAUDE_CODE,
            provider_names::CODEX,
            provider_names::DSH,
        ] {
            assert!(is_builtin_provider(name), "{name} must be a built-in name");
        }
        assert!(!is_builtin_provider("openjiuwen"));
        assert!(!is_builtin_provider(""));
    }

    #[test]
    fn descriptor_serializes_snake_case() {
        let descriptor = ProviderDescriptor {
            name: "native".to_owned(),
            card_name: "openjiuwen-native".to_owned(),
            implementation_version: "0.1.0".to_owned(),
            builtin: true,
            implemented: true,
            configures_models: true,
            requires: None,
        };
        let value = serde_json::to_value(&descriptor).expect("the descriptor serializes");
        assert_eq!(value["card_name"], "openjiuwen-native");
        assert_eq!(value["configures_models"], true);
        assert!(value.get("requires").is_some());
    }
}
