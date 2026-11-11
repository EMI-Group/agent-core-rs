//! Harness identity and capability negotiation.
//!
//! Mirrors `openjiuwen.harness_protocol.models` (v0.1.20): [`HarnessCapability`],
//! [`HostCapability`], [`HarnessCard`], `PROTOCOL_VERSION`, and
//! `default_compatible_protocol_versions`.
//!
//! Python's `frozenset` fields become [`BTreeSet`] (sorted JSON arrays, cheap equality, and
//! `Ord`); the re-freezing in `__post_init__` has no analogue because Rust values are owned and
//! compared by value. Constructors do not validate implicitly — decode paths and callers that
//! need the Python invariants call [`Validate::validate`].

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::errors::{HarnessError, HarnessResult};
use crate::validation::{join_sorted, Validate};

/// Version of the harness protocol implemented by this crate.
pub const PROTOCOL_VERSION: &str = "1.0";

/// Optional behavior an external harness may advertise in its [`HarnessCard`].
///
/// Wire values are `steer`, `graceful_abort`, `force_abort`, `pause_resume`,
/// `persistent_session`, `checkpoint`, `native_tools`, `mcp_tools`, `hooks`, `model_selection`,
/// `model_discovery`, `endpoint_models`, and `auth_fallback`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum HarnessCapability {
    /// A new input can be steered into the turn that is already running.
    Steer,
    /// The active turn can be aborted after it reaches a safe point.
    GracefulAbort,
    /// The active turn can be aborted immediately.
    ForceAbort,
    /// A running turn can be paused and resumed.
    PauseResume,
    /// One harness session survives across turns and process lifetimes.
    PersistentSession,
    /// The harness exports and imports opaque checkpoints.
    Checkpoint,
    /// The harness executes tools itself instead of delegating every call to the host.
    NativeTools,
    /// The harness consumes MCP server configurations from the host.
    McpTools,
    /// The harness dispatches hooks supplied by the host.
    Hooks,
    /// `HarnessModelControl::set_model` can switch the model or effort mid-session.
    ModelSelection,
    /// `HarnessModelControl::list_models` can probe the provider's model catalog.
    ModelDiscovery,
    /// `ModelConfigProvider::model_config` accepts a `ModelChoice` that carries an
    /// `endpoint` (a host-supplied API base or key), not only built-in models.
    EndpointModels,
    /// A built-in model that fails authentication switches the session once to the fallback
    /// model, after the host ratifies the switch.
    AuthFallback,
}

impl HarnessCapability {
    /// Returns the Python `.value` string of this capability.
    ///
    /// The wire string is also needed to reproduce Python's `sorted(capability.value ...)`
    /// ordering in negotiation and validation messages.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Steer => "steer",
            Self::GracefulAbort => "graceful_abort",
            Self::ForceAbort => "force_abort",
            Self::PauseResume => "pause_resume",
            Self::PersistentSession => "persistent_session",
            Self::Checkpoint => "checkpoint",
            Self::NativeTools => "native_tools",
            Self::McpTools => "mcp_tools",
            Self::Hooks => "hooks",
            Self::ModelSelection => "model_selection",
            Self::ModelDiscovery => "model_discovery",
            Self::EndpointModels => "endpoint_models",
            Self::AuthFallback => "auth_fallback",
        }
    }
}

/// Fine-grained host service available to a harness implementation.
///
/// Wire values are `checkpoint_sink`, `native_tool_gateway`, `mcp_servers`, `hooks`,
/// `tool_approval`, `user_input`, `mcp_elicitation`, `dynamic_tool_call`,
/// `provider_interaction`, `telemetry`, and `model_request_observation`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum HostCapability {
    /// The host supplies a checkpoint sink the harness can persist checkpoints into.
    CheckpointSink,
    /// The host executes native tool calls through its tool gateway.
    NativeToolGateway,
    /// The host supplies MCP server configurations.
    McpServers,
    /// The host dispatches hook callbacks for the harness.
    Hooks,
    /// The host can approve or deny tool calls before they run.
    ToolApproval,
    /// The host can answer user-input interactions.
    UserInput,
    /// The host can answer MCP elicitation requests.
    McpElicitation,
    /// The host can execute tools that are discovered dynamically at run time.
    DynamicToolCall,
    /// The host can complete provider-specific interactions.
    ProviderInteraction,
    /// The host records telemetry observations.
    Telemetry,
    /// The host consumes `ModelRequestEvent` observations; providers open their vendor-side
    /// observation channels only when this is declared.
    ModelRequestObservation,
}

impl HostCapability {
    /// Returns the Python `.value` string of this host capability.
    ///
    /// The wire string is also needed to reproduce Python's `sorted(capability.value ...)`
    /// ordering in negotiation and validation messages.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CheckpointSink => "checkpoint_sink",
            Self::NativeToolGateway => "native_tool_gateway",
            Self::McpServers => "mcp_servers",
            Self::Hooks => "hooks",
            Self::ToolApproval => "tool_approval",
            Self::UserInput => "user_input",
            Self::McpElicitation => "mcp_elicitation",
            Self::DynamicToolCall => "dynamic_tool_call",
            Self::ProviderInteraction => "provider_interaction",
            Self::Telemetry => "telemetry",
            Self::ModelRequestObservation => "model_request_observation",
        }
    }
}

/// Static identity and capability metadata for one harness implementation.
///
/// `fork_modes` lists the session-fork modes the provider materialises natively — the documented
/// values are `full` and `before`; an empty set means the provider has no native fork, and a host
/// must not emulate one from a conversation mirror.
///
/// [`HarnessCard::default`] leaves `name` and `implementation_version` empty, so the card is
/// invalid until they are set (use [`HarnessCard::new`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct HarnessCard {
    /// Stable identifier of the harness implementation.
    pub name: String,
    /// Version of the harness implementation itself (independent of the protocol version).
    pub implementation_version: String,
    /// Protocol version the harness implements; defaults to [`PROTOCOL_VERSION`].
    #[serde(default = "default_protocol_version")]
    pub protocol_version: String,
    /// Protocol versions the harness accepts from a host; defaults to the set returned by
    /// [`default_compatible_protocol_versions`].
    #[serde(default = "default_compatible_protocol_versions")]
    pub compatible_protocol_versions: BTreeSet<String>,
    /// Capabilities the harness advertises; defaults to empty.
    #[serde(default)]
    pub capabilities: BTreeSet<HarnessCapability>,
    /// Host capabilities the harness cannot run without; defaults to empty.
    #[serde(default)]
    pub required_host_capabilities: BTreeSet<HostCapability>,
    /// Host capabilities the harness can exploit when present; defaults to empty.
    #[serde(default)]
    pub optional_host_capabilities: BTreeSet<HostCapability>,
    /// Native session-fork modes (`full`, `before`); defaults to empty.
    #[serde(default)]
    pub fork_modes: BTreeSet<String>,
}

impl Default for HarnessCard {
    /// Returns an invalid placeholder card: `name` and `implementation_version` stay empty and
    /// must be set before the card is used.
    fn default() -> Self {
        Self {
            name: String::new(),
            implementation_version: String::new(),
            protocol_version: PROTOCOL_VERSION.to_owned(),
            compatible_protocol_versions: default_compatible_protocol_versions(),
            capabilities: BTreeSet::new(),
            required_host_capabilities: BTreeSet::new(),
            optional_host_capabilities: BTreeSet::new(),
            fork_modes: BTreeSet::new(),
        }
    }
}

/// Returns the protocol version set a new card accepts by default.
///
/// The set contains only [`PROTOCOL_VERSION`].
#[must_use]
pub fn default_compatible_protocol_versions() -> BTreeSet<String> {
    BTreeSet::from([PROTOCOL_VERSION.to_owned()])
}

/// Returns the serde default for [`HarnessCard::protocol_version`].
fn default_protocol_version() -> String {
    PROTOCOL_VERSION.to_owned()
}

impl HarnessCard {
    /// Builds a card for `name`/`implementation_version` with the Python defaults for every
    /// other field.
    #[must_use]
    pub fn new(name: impl Into<String>, implementation_version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            implementation_version: implementation_version.into(),
            ..Self::default()
        }
    }

    /// Returns whether the implementation declares `capability`.
    #[must_use]
    pub fn supports(&self, capability: HarnessCapability) -> bool {
        self.capabilities.contains(&capability)
    }

    /// Fails fast when a host cannot satisfy this provider's requirements.
    ///
    /// The checks run in Python's order: (1) the host protocol version must be listed in
    /// [`HarnessCard::compatible_protocol_versions`]; (2) every
    /// [`HarnessCard::required_host_capabilities`] entry must be present in `capabilities`;
    /// (3) [`HarnessCard::optional_host_capabilities`] is never validated.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Protocol`] with `"host protocol '{version}' is not supported by
    /// '{name}'"`, or `"host is missing required capabilities: {sorted, ", "}"` listing the
    /// missing capabilities by wire value.
    pub fn validate_host(
        &self,
        protocol_version: &str,
        capabilities: &BTreeSet<HostCapability>,
    ) -> HarnessResult<()> {
        if !self.compatible_protocol_versions.contains(protocol_version) {
            return Err(HarnessError::protocol(format!(
                "host protocol '{protocol_version}' is not supported by '{}'",
                self.name
            )));
        }
        let missing: Vec<&str> = self
            .required_host_capabilities
            .difference(capabilities)
            .copied()
            .map(HostCapability::as_str)
            .collect();
        if !missing.is_empty() {
            return Err(HarnessError::protocol(format!(
                "host is missing required capabilities: {}",
                join_sorted(missing)
            )));
        }
        Ok(())
    }
}

impl Validate for HarnessCard {
    /// Checks Python's `__post_init__` invariants in order: non-empty identity/versions, the
    /// implemented protocol version is compatible, and no host capability is both required and
    /// optional.
    fn validate(&self) -> HarnessResult<()> {
        if self.name.is_empty()
            || self.implementation_version.is_empty()
            || self.protocol_version.is_empty()
        {
            return Err(HarnessError::validation(
                "harness card identity and versions must not be empty",
            ));
        }
        if !self
            .compatible_protocol_versions
            .contains(&self.protocol_version)
        {
            return Err(HarnessError::validation(
                "card protocol_version must be included in compatible_protocol_versions",
            ));
        }
        let overlap: BTreeSet<&str> = self
            .required_host_capabilities
            .intersection(&self.optional_host_capabilities)
            .copied()
            .map(HostCapability::as_str)
            .collect();
        if !overlap.is_empty() {
            return Err(HarnessError::validation(format!(
                "host capabilities cannot be both required and optional: {}",
                join_sorted(overlap)
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::collections::BTreeSet;

    use serde_json::json;

    use super::{
        default_compatible_protocol_versions, HarnessCapability, HarnessCard, HostCapability,
        PROTOCOL_VERSION,
    };
    use crate::{from_str_validated, HarnessErrorKind, Validate};

    fn card() -> HarnessCard {
        HarnessCard {
            capabilities: BTreeSet::from([HarnessCapability::Steer, HarnessCapability::Hooks]),
            required_host_capabilities: BTreeSet::from([HostCapability::CheckpointSink]),
            optional_host_capabilities: BTreeSet::from([HostCapability::Telemetry]),
            fork_modes: BTreeSet::from(["full".to_owned()]),
            ..HarnessCard::new("acme-harness", "1.2.3")
        }
    }

    #[test]
    fn defaults_are_the_python_defaults() {
        let defaults = default_compatible_protocol_versions();
        assert_eq!(defaults, BTreeSet::from([PROTOCOL_VERSION.to_owned()]));

        let default_card = HarnessCard::default();
        assert_eq!(default_card.name, "");
        assert_eq!(default_card.implementation_version, "");
        assert_eq!(default_card.protocol_version, PROTOCOL_VERSION);
        assert_eq!(default_card.compatible_protocol_versions, defaults);
        assert!(default_card.capabilities.is_empty());
        assert!(default_card.required_host_capabilities.is_empty());
        assert!(default_card.optional_host_capabilities.is_empty());
        assert!(default_card.fork_modes.is_empty());
        assert_eq!(
            default_card
                .validate()
                .expect_err("identity missing")
                .message(),
            "harness card identity and versions must not be empty"
        );
    }

    #[test]
    fn supports_reports_declared_capabilities_only() {
        let card = card();
        assert!(card.supports(HarnessCapability::Steer));
        assert!(card.supports(HarnessCapability::Hooks));
        assert!(!card.supports(HarnessCapability::ForceAbort));
        assert!(!card.supports(HarnessCapability::AuthFallback));
    }

    #[test]
    fn validate_host_accepts_supported_version_and_satisfied_requirements() {
        let card = card();
        let host = BTreeSet::from([HostCapability::CheckpointSink]);
        assert!(card.validate_host(PROTOCOL_VERSION, &host).is_ok());
        // Extra host capabilities are irrelevant, and optional ones need not be present.
        let generous_host = BTreeSet::from([
            HostCapability::CheckpointSink,
            HostCapability::Telemetry,
            HostCapability::Hooks,
        ]);
        assert!(card.validate_host(PROTOCOL_VERSION, &generous_host).is_ok());
    }

    #[test]
    fn validate_host_rejects_unsupported_versions() {
        let error = card()
            .validate_host("9.9", &BTreeSet::from([HostCapability::CheckpointSink]))
            .expect_err("unsupported version");
        assert_eq!(error.kind(), HarnessErrorKind::Protocol);
        assert_eq!(
            error.message(),
            "host protocol '9.9' is not supported by 'acme-harness'"
        );

        let mut legacy = card();
        legacy.compatible_protocol_versions.insert("0.9".to_owned());
        assert!(legacy
            .validate_host("0.9", &BTreeSet::from([HostCapability::CheckpointSink]))
            .is_ok());
    }

    #[test]
    fn validate_host_lists_missing_capabilities_sorted() {
        let mut card = card();
        card.required_host_capabilities
            .insert(HostCapability::Telemetry);
        card.required_host_capabilities
            .insert(HostCapability::NativeToolGateway);
        let error = card
            .validate_host(PROTOCOL_VERSION, &BTreeSet::new())
            .expect_err("missing capabilities");
        assert_eq!(error.kind(), HarnessErrorKind::Protocol);
        assert_eq!(
            error.message(),
            "host is missing required capabilities: checkpoint_sink, native_tool_gateway, telemetry"
        );
    }

    #[test]
    fn validate_host_ignores_optional_capabilities() {
        let mut card = card();
        card.required_host_capabilities.clear();
        card.optional_host_capabilities
            .insert(HostCapability::UserInput);
        assert!(card
            .validate_host(PROTOCOL_VERSION, &BTreeSet::new())
            .is_ok());
    }

    #[test]
    fn card_self_consistency_rules() {
        assert!(card().validate().is_ok());

        let blank_version = HarnessCard {
            protocol_version: String::new(),
            ..card()
        };
        assert_eq!(
            blank_version
                .validate()
                .expect_err("blank version")
                .message(),
            "harness card identity and versions must not be empty"
        );

        let incompatible = HarnessCard {
            compatible_protocol_versions: BTreeSet::from(["0.9".to_owned()]),
            ..card()
        };
        assert_eq!(
            incompatible.validate().expect_err("incompatible").message(),
            "card protocol_version must be included in compatible_protocol_versions"
        );

        let overlap = HarnessCard {
            required_host_capabilities: BTreeSet::from([
                HostCapability::CheckpointSink,
                HostCapability::Telemetry,
            ]),
            optional_host_capabilities: BTreeSet::from([
                HostCapability::Telemetry,
                HostCapability::CheckpointSink,
            ]),
            ..card()
        };
        assert_eq!(
            overlap.validate().expect_err("overlap").message(),
            "host capabilities cannot be both required and optional: checkpoint_sink, telemetry"
        );
    }

    #[test]
    fn decode_applies_defaults_and_validates() {
        let decoded: HarnessCard =
            from_str_validated(r#"{"name": "acme", "implementation_version": "1.2.3"}"#)
                .expect("valid card");
        assert_eq!(decoded.protocol_version, PROTOCOL_VERSION);
        assert_eq!(
            decoded.compatible_protocol_versions,
            default_compatible_protocol_versions()
        );
        assert!(decoded.capabilities.is_empty());
        assert!(decoded.fork_modes.is_empty());
        decoded.validate().expect("valid card");

        let error = from_str_validated::<HarnessCard>(r#"{"implementation_version": "1.2.3"}"#)
            .expect_err("missing name");
        assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    }

    #[test]
    fn enum_wire_values_match_python() {
        for (capability, wire) in [
            (HarnessCapability::Steer, "steer"),
            (HarnessCapability::PauseResume, "pause_resume"),
            (HarnessCapability::ModelSelection, "model_selection"),
            (HarnessCapability::ModelDiscovery, "model_discovery"),
            (HarnessCapability::EndpointModels, "endpoint_models"),
            (HarnessCapability::AuthFallback, "auth_fallback"),
        ] {
            assert_eq!(capability.as_str(), wire);
            assert_eq!(
                serde_json::to_value(capability).expect("serializes"),
                json!(wire)
            );
        }
        for (capability, wire) in [
            (HostCapability::CheckpointSink, "checkpoint_sink"),
            (HostCapability::McpElicitation, "mcp_elicitation"),
            (HostCapability::ProviderInteraction, "provider_interaction"),
            (
                HostCapability::ModelRequestObservation,
                "model_request_observation",
            ),
        ] {
            assert_eq!(capability.as_str(), wire);
            assert_eq!(
                serde_json::to_value(capability).expect("serializes"),
                json!(wire)
            );
        }
    }

    #[test]
    fn card_round_trips_with_every_key() {
        let card = card();
        let encoded = serde_json::to_value(&card).expect("serializes");
        let keys: Vec<&str> = encoded
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            [
                "name",
                "implementation_version",
                "protocol_version",
                "compatible_protocol_versions",
                "capabilities",
                "required_host_capabilities",
                "optional_host_capabilities",
                "fork_modes",
            ]
        );
        assert_eq!(encoded["fork_modes"], json!(["full"]));
        // `BTreeSet` order is the enum declaration order, so the array is deterministic.
        assert_eq!(encoded["capabilities"], json!(["steer", "hooks"]));
        let decoded: HarnessCard = from_str_validated(&encoded.to_string()).expect("round trip");
        assert_eq!(decoded, card);
    }
}
