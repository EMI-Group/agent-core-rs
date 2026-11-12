//! The native openJiuwen provider: wiring over `openjiuwen-harness` only.
//!
//! [`provider`] returns the harness crate's [`NativeHarnessProvider`] and [`card`] its capability
//! card; this module adds the registry identity (`PROVIDER_NAME`) and the card name, and nothing
//! else. Harness behavior lives in `openjiuwen-harness` and is never reimplemented here.
//!
//! # Python parity deviations
//!
//! - Registry name and card name differ: the registry key is `native` while the card reports
//!   `openjiuwen-native`; Python registers `native` with the card name `deepagent`.
//! - The Rust native provider **is** a [`ModelConfigProvider`](openjiuwen_harness_protocol::ModelConfigProvider)
//!   (it maps a reasoning effort onto the create `temperature`), so it registers through
//!   [`model_configuring_provider`](crate::model_configuring_provider). Python's `native` provider
//!   is not a `ModelConfigProvider`.
//! - The accepted create keys are the Rust set — `model`, `provider`, `api_key`, `api_base`,
//!   `temperature`, `workspace`, `enable_task_loop`, `max_iterations`, `system_prompt`,
//!   `permissions`, `language`. Python's `deep_agent` / `agent_template` / `session_id` /
//!   `event_buffer_capacity` keys are not accepted: the Rust `DeepAgent` takes live handles through
//!   its builder instead of an agent-template spec tree.
use openjiuwen_harness::protocol::{native_card, HARNESS_NAME};
use openjiuwen_harness_protocol::HarnessCard;

pub use openjiuwen_harness::protocol::PROVIDER_VERSION;
pub use openjiuwen_harness::{NativeHarness, NativeHarnessOptions, NativeHarnessProvider};

/// Registry name of the native provider.
pub const PROVIDER_NAME: &str = "native";

/// Card name the native provider reports (`openjiuwen-native`, Python `deepagent`).
pub const CARD_NAME: &str = HARNESS_NAME;

/// Returns the native harness provider (`NativeHarnessProvider::new()`).
#[must_use]
pub fn provider() -> NativeHarnessProvider {
    NativeHarnessProvider::new()
}

/// Returns the capability card the native provider reports.
///
/// The card is [`native_card`] evaluated with the crate's own name and version, so it matches
/// [`NativeHarnessProvider::card`] exactly.
#[must_use]
pub fn card() -> HarnessCard {
    native_card(CARD_NAME, PROVIDER_VERSION)
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use openjiuwen_harness::protocol::{native_card, HARNESS_NAME, PROVIDER_VERSION};
    use openjiuwen_harness_protocol::{HarnessCapability, HarnessProvider, HostCapability};

    use super::{card, provider, CARD_NAME, PROVIDER_NAME};

    #[test]
    fn card_name_is_the_harness_name() {
        assert_eq!(CARD_NAME, "openjiuwen-native");
        assert_eq!(CARD_NAME, HARNESS_NAME);
        assert_eq!(PROVIDER_NAME, "native");
    }

    #[test]
    fn card_matches_the_harness_card() {
        let expected = native_card(HARNESS_NAME, PROVIDER_VERSION);
        assert_eq!(card(), expected);
    }

    #[test]
    fn card_declares_the_native_capabilities() {
        let card = card();
        assert_eq!(card.implementation_version, PROVIDER_VERSION);
        for capability in [
            HarnessCapability::Steer,
            HarnessCapability::GracefulAbort,
            HarnessCapability::ForceAbort,
            HarnessCapability::PauseResume,
            HarnessCapability::PersistentSession,
            HarnessCapability::Checkpoint,
            HarnessCapability::NativeTools,
            HarnessCapability::McpTools,
            HarnessCapability::ModelSelection,
            HarnessCapability::ModelDiscovery,
        ] {
            assert!(card.supports(capability), "{capability:?} must be declared");
        }
        assert!(card.required_host_capabilities.is_empty());
        for capability in [
            HostCapability::CheckpointSink,
            HostCapability::ToolApproval,
            HostCapability::UserInput,
            HostCapability::Telemetry,
        ] {
            assert!(
                card.optional_host_capabilities.contains(&capability),
                "{capability:?} must be optional-host"
            );
        }
        assert!(card.fork_modes.is_empty());
    }

    #[test]
    fn provider_reports_the_shared_card() {
        assert_eq!(provider().card(), &card());
    }
}
