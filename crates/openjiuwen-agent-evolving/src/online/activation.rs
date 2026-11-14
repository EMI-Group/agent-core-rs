//! The activation seam: how a finished training run's artifact becomes the served policy.
//!
//! Python parity: `agent_rl/online/training_runner.py`'s `LoRAActivator` (the
//! training runner calls it once the artifact is exported). Python reaches the
//! vendor AI gateway (AIGW) to swap the served `LoRA`; the port keeps the seam and
//! ships two implementations: [`NoopLoraActivator`], which records the request in
//! the [`OnlineStore`] so tests and local runs stay deterministic, and
//! [`UnsupportedLoraActivator`], which answers with a typed
//! [`Error::Unsupported`] naming the
//! gateway that is missing.
//!
//! Both are idempotent: activating the same artifact twice succeeds, which is what
//! lets [`TrainingRunner::recover`](crate::online::TrainingRunner::recover) retry a
//! run that crashed after exporting its artifact.

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result as KernelResult};
use serde::{Deserialize, Serialize};

use super::store::{keys, OnlineStore};

/// Activates the artifact a training run exported.
#[async_trait]
pub trait LoraActivator: Send + Sync {
    /// Activates (or idempotently confirms) `request`'s artifact.
    async fn activate(&self, request: &LoraActivation) -> KernelResult<()>;
}

/// One activation request handed to a [`LoraActivator`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LoraActivation {
    /// Model identity the artifact belongs to.
    pub base_model: String,
    /// Artifact name to activate.
    pub lora_name: String,
    /// Artifact path to activate.
    pub lora_path: String,
    /// Policy name that must currently be active (the run's parent policy).
    pub expected_lora_name: String,
    /// Run the activation belongs to.
    pub training_run_id: String,
}

/// A [`LoraActivator`] that records the activation instead of performing it.
///
/// It writes the request under `rl:v1:training_run:{id}:activation`, which makes
/// activation observable and deterministic for tests and local runs without a
/// model server.
#[derive(Debug, Clone)]
pub struct NoopLoraActivator<S: OnlineStore> {
    store: S,
}

impl<S: OnlineStore> NoopLoraActivator<S> {
    /// Creates the activator over `store`.
    pub fn new(store: S) -> Self {
        Self { store }
    }

    /// The backing store.
    #[must_use]
    pub fn store(&self) -> &S {
        &self.store
    }

    /// The activation recorded for `training_run_id`, if any.
    pub async fn recorded(&self, training_run_id: &str) -> KernelResult<Option<LoraActivation>> {
        match self
            .store
            .get(&keys::training_activation(training_run_id))
            .await?
        {
            Some(payload) => Ok(Some(
                serde_json::from_str(&payload).map_err(Error::external)?,
            )),
            None => Ok(None),
        }
    }
}

#[async_trait]
impl<S: OnlineStore> LoraActivator for NoopLoraActivator<S> {
    async fn activate(&self, request: &LoraActivation) -> KernelResult<()> {
        let encoded = serde_json::to_string(request).map_err(Error::external)?;
        self.store
            .set(
                &keys::training_activation(&request.training_run_id),
                &encoded,
            )
            .await
    }
}

/// The vendor activation path, retained as a typed seam.
///
/// Python calls the AI gateway (AIGW) to swap the served `LoRA`; that transport is
/// vendor HTTP and is not ported, so every activation fails with a reason that
/// names it. The artifact is still exported and recorded — only the switch is
/// missing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedLoraActivator {
    gateway: String,
}

impl UnsupportedLoraActivator {
    /// Creates the activator for the default gateway name.
    #[must_use]
    pub fn new() -> Self {
        Self::with_gateway("vendor AI gateway (AIGW)")
    }

    /// Creates the activator for an explicitly named gateway.
    #[must_use]
    pub fn with_gateway(gateway: impl Into<String>) -> Self {
        Self {
            gateway: gateway.into(),
        }
    }

    /// The gateway this activator would have called.
    #[must_use]
    pub fn gateway(&self) -> &str {
        &self.gateway
    }
}

impl Default for UnsupportedLoraActivator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LoraActivator for UnsupportedLoraActivator {
    async fn activate(&self, request: &LoraActivation) -> KernelResult<()> {
        Err(Error::unsupported(format!(
            "LoRA activation is not available in this build: switching {} requires the {} HTTP \
             service (run {})",
            request.lora_name, self.gateway, request.training_run_id
        )))
    }
}
