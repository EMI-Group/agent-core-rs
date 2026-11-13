//! The `native_v2` provider: an agent-teams-hosted harness behind the shared vendor shell.
//!
//! The provider is the Rust counterpart of Python's `NativeHarnessProtocolAdapter`: it does not
//! implement a harness of its own. `NativeV2Driver` hosts an `openjiuwen-agent-teams` team
//! (`TeamRuntimeManager::activate`) and drives it turn by turn, while
//! [`VendorHarness`](crate::vendor_cli::VendorHarness) supplies the whole
//! [`HarnessProtocol`](openjiuwen_harness_protocol::HarnessProtocol) surface: lifecycle, turn
//! queue, event sequence, capability gating and teardown.
//!
//! # Feature gating
//!
//! The module exists only with the `native-v2` feature, which pulls in the optional
//! `openjiuwen-agent-teams` dependency. Without it
//! [`crate::providers::builtin_provider`] reports the missing feature instead of resolving the
//! name, so a build never pays for the team runtime it does not use.
//!
//! # Create schema
//!
//! `NativeV2Config` is the provider's create mapping. Unlike Python's adapter it carries no
//! `deep_agent` or `agent_template` tree: the Rust team runtime takes a
//! [`TeamAgentSpec`](openjiuwen_agent_teams::schema::TeamAgentSpec) directly, and members get
//! their harnesses from the injected
//! [`MemberHarnessFactory`](openjiuwen_agent_teams::harness::MemberHarnessFactory).
//!
//! # Card
//!
//! `card` declares only what the driver really does through the team runtime: steering, both
//! abort modes, pause/resume, the persistent team session and native tools. Checkpoints, MCP
//! tools and model selection are deliberately absent — this adapter has no portable snapshot and
//! never reconfigures a member's model.
use std::sync::Arc;

use openjiuwen_agent_teams::schema::TeamAgentSpec;
use openjiuwen_common::error::Error;
use openjiuwen_harness_protocol::{
    EventBufferConfig, HarnessCapability, HarnessCard, HarnessError, HarnessProtocol,
    HarnessProvider, HarnessResult, HostCapability, JsonObject, JsonValue,
};
use serde::{Deserialize, Serialize};

use crate::decode::{expect_object, optional_str, optional_u64, reject_unknown_keys};
use crate::vendor_cli::VendorHarness;

mod driver;

pub use driver::NativeV2Driver;

/// Registry name of the `native_v2` provider.
pub const PROVIDER_NAME: &str = "native_v2";

/// Card name the `native_v2` provider reports.
pub const CARD_NAME: &str = "native_v2";

/// Version of the adapter whose card is reported here (Python `NativeHarnessProtocolAdapter`).
pub const IMPLEMENTATION_VERSION: &str = "0.2.0";

/// Default capacity of the observation buffer the created harness advertises.
pub const DEFAULT_EVENT_BUFFER_CAPACITY: u64 = 1024;

/// Configuration keys [`NativeV2Config::from_json`] accepts.
const CONFIG_FIELDS: [&str; 4] = ["team", "session_id", "language", "event_buffer_capacity"];

/// Create mapping of the `native_v2` provider.
///
/// `team` is mandatory; the other three keys have defaults. The plain serde decoder tolerates
/// unknown keys (the crate-wide rule) while [`NativeV2Config::from_json`] — the provider SPI path
/// — rejects them like Python's `from_mapping`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct NativeV2Config {
    /// Blueprint of the team the harness hosts; validated with
    /// [`TeamAgentSpec::validate`](openjiuwen_agent_teams::schema::TeamAgentSpec::validate).
    pub team: TeamAgentSpec,
    /// Activation session; `None` uses the host session id of the `start` call.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Team language (`cn`/`cf`/`en`); `None` keeps the runtime default.
    #[serde(default)]
    pub language: Option<String>,
    /// Capacity of the observation buffer the harness advertises.
    #[serde(default = "default_event_buffer_capacity")]
    pub event_buffer_capacity: u64,
}

impl NativeV2Config {
    /// Decodes a provider SPI mapping, validating it like Python's `from_mapping`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unknown key
    /// (`"unknown native_v2 configuration fields: …"`), a missing or non-object `team`
    /// (`"native_v2 requires a team specification"`), a blueprint whose own
    /// [`validate`](openjiuwen_agent_teams::schema::TeamAgentSpec::validate) fails, a wrongly
    /// typed `session_id`/`language`/`event_buffer_capacity`, an empty `session_id` or `language`,
    /// or a zero capacity.
    pub fn from_json(object: &JsonObject) -> Result<Self, Error> {
        reject_unknown_keys(
            object,
            &CONFIG_FIELDS,
            "unknown native_v2 configuration fields",
        )?;
        let team = decode_team(object.get("team"))?;
        let session_id = optional_str(object, "session_id")?;
        let language = optional_str(object, "language")?;
        for (key, value) in [("session_id", &session_id), ("language", &language)] {
            if value.as_deref().is_some_and(|text| text.trim().is_empty()) {
                return Err(Error::validation(format!(
                    "native_v2 {key} must not be empty"
                )));
            }
        }
        let event_buffer_capacity =
            optional_u64(object, "event_buffer_capacity")?.unwrap_or(DEFAULT_EVENT_BUFFER_CAPACITY);
        if event_buffer_capacity == 0 {
            return Err(Error::validation(
                "native_v2 event_buffer_capacity must be positive",
            ));
        }
        Ok(Self {
            team,
            session_id,
            language,
            event_buffer_capacity,
        })
    }

    /// Returns the observation-buffer contract of the created harness.
    #[must_use]
    pub fn event_buffer_config(&self) -> EventBufferConfig {
        EventBufferConfig {
            capacity: self.event_buffer_capacity,
            ..EventBufferConfig::default()
        }
    }
}

/// Decodes the mandatory `team` key into a validated blueprint.
///
/// # Errors
///
/// Returns [`Error::Validation`] when the key is absent, not an object, not decodable as a
/// [`TeamAgentSpec`], or rejected by the blueprint's own validation.
fn decode_team(value: Option<&JsonValue>) -> Result<TeamAgentSpec, Error> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Err(Error::validation("native_v2 requires a team specification"));
    };
    let object = expect_object(value, "team")?;
    let map: serde_json::Map<String, JsonValue> = object.into_iter().collect();
    let team: TeamAgentSpec = serde_json::from_value(JsonValue::Object(map))
        .map_err(|error| Error::validation(format!("team is not a team specification: {error}")))?;
    team.validate()?;
    Ok(team)
}

/// Returns [`DEFAULT_EVENT_BUFFER_CAPACITY`].
const fn default_event_buffer_capacity() -> u64 {
    DEFAULT_EVENT_BUFFER_CAPACITY
}

/// Returns the capability card of the `native_v2` adapter.
///
/// The declared set is exactly what the driver reaches through `TeamAgent::runtime()` and
/// `TeamAgent::stop`: steering, graceful and forced abort, pause/resume, the persistent team
/// session, and native tools (the team's members run their own tools).
#[must_use]
pub fn card() -> HarnessCard {
    let mut card = HarnessCard::new(CARD_NAME, IMPLEMENTATION_VERSION);
    card.capabilities = [
        HarnessCapability::PersistentSession,
        HarnessCapability::GracefulAbort,
        HarnessCapability::ForceAbort,
        HarnessCapability::PauseResume,
        HarnessCapability::Steer,
        HarnessCapability::NativeTools,
    ]
    .into_iter()
    .collect();
    card.optional_host_capabilities = [HostCapability::UserInput, HostCapability::CheckpointSink]
        .into_iter()
        .collect();
    card
}

/// Returns the `native_v2` provider (`NativeV2HarnessProvider::default()`).
#[must_use]
pub fn provider() -> NativeV2HarnessProvider {
    NativeV2HarnessProvider::default()
}

/// The `native_v2` provider: identity card plus a `create` that builds an unstarted
/// [`VendorHarness`].
///
/// The provider is deliberately **not** a
/// [`ModelConfigProvider`](openjiuwen_harness_protocol::ModelConfigProvider): a team member's
/// model comes from the injected harness factory, never from the create mapping.
#[derive(Debug, Clone)]
pub struct NativeV2HarnessProvider {
    /// Card reported by [`HarnessProvider::card`].
    card: HarnessCard,
}

impl Default for NativeV2HarnessProvider {
    /// Builds the provider with the card of [`card`].
    fn default() -> Self {
        Self { card: card() }
    }
}

impl HarnessProvider for NativeV2HarnessProvider {
    /// Returns the `native_v2` adapter card.
    fn card(&self) -> &HarnessCard {
        &self.card
    }

    /// Validates `config` and builds an unstarted harness over a [`NativeV2Driver`].
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Validation`] when `config` is not a valid [`NativeV2Config`].
    fn create(&self, config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>> {
        let config = NativeV2Config::from_json(&config).map_err(config_error_to_harness)?;
        let buffer = config.event_buffer_config();
        let driver = NativeV2Driver::new(config);
        Ok(Box::new(VendorHarness::new(
            card(),
            buffer,
            Arc::new(driver),
        )))
    }
}

/// Translates a configuration error into the harness SPI error the provider reports.
///
/// [`Error::Validation`] keeps its message verbatim; every other kind is a configuration-shape
/// problem too, so it joins the same [`HarnessError::Validation`] family with its rendered text
/// (which carries the workspace error prefix).
fn config_error_to_harness(error: Error) -> HarnessError {
    match error {
        Error::Validation(message) => HarnessError::validation(message),
        other => HarnessError::validation(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use openjiuwen_harness_protocol::{
        HarnessCapability, HarnessErrorKind, HarnessProvider, HarnessState, HostCapability,
        PROTOCOL_VERSION,
    };
    use serde_json::json;

    use super::{
        card, provider, CARD_NAME, DEFAULT_EVENT_BUFFER_CAPACITY, IMPLEMENTATION_VERSION,
        PROVIDER_NAME,
    };

    /// Returns a minimal valid team blueprint as JSON.
    fn team_json() -> serde_json::Value {
        json!({
            "team_name": "alpha",
            "agents": {
                "leader": {"card": {"name": "Leader"}, "system_prompt": "lead the team"},
                "teammate": {"card": {"name": "Teammate"}, "system_prompt": "do the work"},
            },
        })
    }

    /// Builds the create mapping with `team` plus the caller's extra keys.
    fn config(extra: &serde_json::Value) -> openjiuwen_harness_protocol::JsonObject {
        let mut object = openjiuwen_harness_protocol::JsonObject::new();
        object.insert("team".to_owned(), team_json());
        for (key, value) in extra.as_object().cloned().unwrap_or_default() {
            object.insert(key, value);
        }
        object
    }

    #[test]
    fn names_and_versions_match_the_python_adapter() {
        assert_eq!(PROVIDER_NAME, "native_v2");
        assert_eq!(CARD_NAME, "native_v2");
        assert_eq!(IMPLEMENTATION_VERSION, "0.2.0");
        assert_eq!(DEFAULT_EVENT_BUFFER_CAPACITY, 1024);
    }

    #[test]
    fn card_declares_exactly_the_implemented_capabilities() {
        let card = card();
        assert_eq!(card.name, CARD_NAME);
        assert_eq!(card.implementation_version, IMPLEMENTATION_VERSION);
        assert_eq!(card.protocol_version, PROTOCOL_VERSION);
        assert_eq!(
            card.capabilities,
            [
                HarnessCapability::PersistentSession,
                HarnessCapability::GracefulAbort,
                HarnessCapability::ForceAbort,
                HarnessCapability::PauseResume,
                HarnessCapability::Steer,
                HarnessCapability::NativeTools,
            ]
            .into_iter()
            .collect()
        );
        for absent in [
            HarnessCapability::Checkpoint,
            HarnessCapability::McpTools,
            HarnessCapability::ModelSelection,
            HarnessCapability::ModelDiscovery,
            HarnessCapability::EndpointModels,
            HarnessCapability::AuthFallback,
            HarnessCapability::Hooks,
        ] {
            assert!(!card.supports(absent), "{absent:?} must not be declared");
        }
        assert_eq!(
            card.optional_host_capabilities,
            [HostCapability::UserInput, HostCapability::CheckpointSink]
                .into_iter()
                .collect()
        );
        assert!(card.required_host_capabilities.is_empty());
        assert!(card.fork_modes.is_empty());
    }

    #[test]
    fn provider_reports_the_shared_card_and_no_model_hook() {
        let provider = provider();
        assert_eq!(provider.card(), &card());
        assert_eq!(provider.card().name, "native_v2");
    }

    #[test]
    fn create_builds_an_unstarted_harness() {
        let harness = provider()
            .create(config(&json!({})))
            .expect("the configuration is valid");
        assert_eq!(harness.state(), HarnessState::Terminated);
        assert_eq!(harness.card().name, CARD_NAME);
        assert_eq!(
            harness.event_buffer_config().capacity,
            DEFAULT_EVENT_BUFFER_CAPACITY
        );
        assert!(harness.provider_session_id().is_none());
    }

    #[test]
    fn create_honours_the_event_buffer_capacity() {
        let harness = provider()
            .create(config(&json!({"event_buffer_capacity": 8})))
            .expect("the configuration is valid");
        assert_eq!(harness.event_buffer_config().capacity, 8);
    }

    #[test]
    fn create_requires_a_team_specification() {
        let error = provider()
            .create(openjiuwen_harness_protocol::JsonObject::new())
            .err()
            .expect("a team is mandatory");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(error.message(), "native_v2 requires a team specification");
    }

    #[test]
    fn create_rejects_unknown_keys() {
        let error = provider()
            .create(config(&json!({"deep_agent": {}})))
            .err()
            .expect("unknown keys are rejected");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(
            error.message(),
            "unknown native_v2 configuration fields: deep_agent"
        );
    }

    #[test]
    fn create_rejects_a_non_object_team_and_an_invalid_blueprint() {
        let mut object = openjiuwen_harness_protocol::JsonObject::new();
        object.insert("team".to_owned(), json!("alpha"));
        let error = provider()
            .create(object)
            .err()
            .expect("team must be an object");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(error.message(), "team must be a JSON object");

        let mut object = openjiuwen_harness_protocol::JsonObject::new();
        object.insert("team".to_owned(), json!({"team_name": "alpha"}));
        let error = provider()
            .create(object)
            .err()
            .expect("a blueprint without a leader is rejected");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(error.message(), "agents dict must contain a 'leader' key");
    }

    #[test]
    fn create_rejects_a_zero_capacity_and_blank_strings() {
        for extra in [
            json!({"event_buffer_capacity": 0}),
            json!({"session_id": " "}),
            json!({"language": ""}),
        ] {
            let error = provider()
                .create(config(&extra))
                .err()
                .expect("the configuration is invalid");
            assert_eq!(error.kind(), HarnessErrorKind::Validation, "{extra}");
        }
    }
}
