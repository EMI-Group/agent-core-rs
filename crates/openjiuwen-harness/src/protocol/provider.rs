//! Native harness provider: create-config parsing and model-choice mapping.
//!
//! Skeleton placeholder: the full implementation lands in this file.

use openjiuwen_harness_protocol::{
    HarnessCard, HarnessError, HarnessProtocol, HarnessProvider, HarnessResult, JsonObject,
    ModelChoice, ModelConfigProvider,
};

use crate::protocol::{native_card, NativeHarnessOptions, HARNESS_NAME, PROVIDER_VERSION};

/// Factory for native harness instances, driven by a JSON create mapping.
#[derive(Debug, Clone)]
pub struct NativeHarnessProvider {
    /// Card shared by every harness this provider creates.
    card: HarnessCard,
    /// Options applied to every created harness.
    options: NativeHarnessOptions,
}

impl Default for NativeHarnessProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeHarnessProvider {
    /// Builds a provider with the default harness options.
    #[must_use]
    pub fn new() -> Self {
        Self {
            card: native_card(HARNESS_NAME, PROVIDER_VERSION),
            options: NativeHarnessOptions::new(),
        }
    }

    /// Replaces the options applied to every created harness.
    #[must_use]
    pub fn with_options(mut self, options: NativeHarnessOptions) -> Self {
        self.options = options;
        self
    }
}

impl HarnessProvider for NativeHarnessProvider {
    fn card(&self) -> &HarnessCard {
        &self.card
    }

    fn create(&self, _config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>> {
        Err(HarnessError::unsupported_capability(
            "native harness provider is not implemented yet",
        ))
    }
}

impl ModelConfigProvider for NativeHarnessProvider {
    fn model_config(
        &self,
        _choice: &ModelChoice,
        _fallback: Option<&ModelChoice>,
    ) -> HarnessResult<JsonObject> {
        Ok(JsonObject::new())
    }
}
