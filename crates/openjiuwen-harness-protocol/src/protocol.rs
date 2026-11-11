//! Behavioral protocol for third-party agent harness integrations.
//!
//! Mirrors `openjiuwen.harness_protocol.protocol` (v0.1.20): the trait seams a host uses to drive
//! an external agent harness — [`HarnessProtocol`] (behavior), [`HarnessModelControl`] (optional
//! model control), [`HarnessProvider`] (factory SPI), and [`ModelConfigProvider`] (model-choice to
//! provider-configuration mapping) — plus the [`DynHarness`], [`DynHarnessProvider`], and
//! [`DynModelControl`] shared-pointer aliases.
//!
//! Python expressed these seams as `typing.Protocol` classes combined with `@runtime_checkable`
//! `isinstance` checks. Rust traits with `Send + Sync` supertraits provide the same guarantee at
//! compile time (`DESIGN.md` §7.4 and §7.10), so the runtime checks have no analogue: hand out a
//! trait object and let the type system enforce conformance.
//!
//! Every harness method takes `&self`, never `&mut self`: implementations synchronize their own
//! state internally, so an `Arc<dyn HarnessProtocol>` is freely shareable and commands may be
//! issued from different tasks — exactly like Python's freely shared instance.
//!
//! Object safety matters: these traits are used as `Arc<dyn …>`/`Box<dyn …>` everywhere, so no
//! generic method and no associated type is part of the design.

use std::sync::Arc;

use async_trait::async_trait;

use crate::checkpoints::HarnessCheckpoint;
use crate::errors::HarnessResult;
use crate::events::EventBufferConfig;
use crate::json::JsonObject;
use crate::models::{
    AbortMode, DeliveryMode, HarnessCard, HarnessContext, HarnessInput, ModelChoice, ModelOption,
    ModelSelection, SendReceipt,
};
use crate::state::HarnessState;
use crate::stream::HarnessEventCursor;

/// Concurrent-safe behavior required of a third-party agent harness.
///
/// One implementation owns one `start`/`stop` cycle at a time. [`events`](Self::events) and
/// [`turn_events`](Self::turn_events) are alternative views over one logical single-consumer
/// observation channel; implementations serialize state transitions internally, so callers may
/// issue commands from different tasks.
///
/// # Shared access
///
/// All methods take `&self` because implementations synchronize their own state internally. That
/// keeps [`DynHarness`] (`Arc<dyn HarnessProtocol>`) freely shareable between the host's runtime
/// tasks, matching Python's freely shared harness instance.
///
/// # Object safety
///
/// The trait is object-safe and intended for `Arc<dyn HarnessProtocol>` / `Box<dyn
/// HarnessProtocol>`; [`HarnessProvider::create`] returns the boxed form.
#[async_trait]
// DESIGN-fixed signatures take owned inputs; do not add clones to satisfy the lint.
#[allow(clippy::needless_pass_by_value)]
pub trait HarnessProtocol: Send + Sync {
    /// Returns the static implementation identity and declared capabilities.
    #[must_use]
    fn card(&self) -> &HarnessCard;

    /// Returns the current high-level lifecycle state.
    #[must_use]
    fn state(&self) -> HarnessState;

    /// Returns the provider-native session id, once one is available.
    #[must_use]
    fn provider_session_id(&self) -> Option<&str>;

    /// Returns the bounded backpressure policy used for observations.
    #[must_use]
    fn event_buffer_config(&self) -> &EventBufferConfig;

    /// Validates host compatibility against the card, starts a cycle, and settles in
    /// [`HarnessState::Idle`].
    ///
    /// The implementation checks [`HarnessContext::protocol_version`] and
    /// [`HarnessContext::host_capabilities`] against the card through
    /// [`HarnessCard::validate_host`], applies [`HarnessContext::resume_policy`] to
    /// [`HarnessContext::checkpoint`], opens a fresh observation cycle for
    /// [`events`](Self::events), and reports [`HarnessState::Idle`] once it accepts input.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Protocol`](crate::HarnessError::Protocol) when the host protocol
    /// version is not compatible with the card or a required host capability is missing, and
    /// [`HarnessError::State`](crate::HarnessError::State) when a cycle cannot be started (for
    /// example because one already runs).
    async fn start(&self, context: HarnessContext) -> HarnessResult<()>;

    /// Stops the cycle, closes the observation channel, and settles in
    /// [`HarnessState::Terminated`].
    ///
    /// The operation is idempotent: stopping an already-terminated harness succeeds without side
    /// effects, and cursors obtained earlier yield `None` instead of new events.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::State`](crate::HarnessError::State) when the provider cannot close
    /// its cycle cleanly, and [`HarnessError::Protocol`](crate::HarnessError::Protocol) when the
    /// provider fails while settling the session.
    async fn stop(&self) -> HarnessResult<()>;

    /// Returns the cycle-long ordered observation stream.
    ///
    /// The cursor stays open across turn boundaries and ends only when the current `start`/`stop`
    /// cycle closes. It shares one logical single-consumer channel with
    /// [`turn_events`](Self::turn_events). A provider must raise
    /// [`HarnessError::State`](crate::HarnessError::State) from the cursor when another observation
    /// cursor is already active instead of racing it; choosing the single-consumer rule over a
    /// panic keeps the error observable by the host.
    #[must_use]
    fn events(&self) -> Box<dyn HarnessEventCursor>;

    /// Returns one finite turn from the observation stream.
    ///
    /// `turn_id` identifies and validates the next unconsumed accepted turn; it is not an
    /// out-of-order selector, and an implementation must not discard intervening turns while
    /// searching for it. When `turn_id` is `None` the cursor uses the next turn. The cursor yields
    /// the turn's `Started` event and the following ordered events through the matching
    /// `Finished`/`Aborted`/`Failed` event; `Paused` and `Resumed` do not end the view.
    ///
    /// This is a serialized convenience view over the same logical channel as
    /// [`events`](Self::events); the two must not be consumed concurrently, and a provider must
    /// raise [`HarnessError::State`](crate::HarnessError::State) for a second concurrently active
    /// cursor rather than racing it.
    #[must_use]
    fn turn_events(&self, turn_id: Option<&str>) -> Box<dyn HarnessEventCursor>;

    /// Accepts input and returns its message-to-turn association without waiting for the turn to
    /// finish.
    ///
    /// A queued (follow-up) input receives its future turn id at acceptance; a steering input
    /// receives the active turn id. `mode` selects the delivery semantics, and
    /// [`DeliveryMode::Auto`] lets the harness choose based on its current state.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::UnsupportedCapability`](crate::HarnessError::UnsupportedCapability)
    /// when the requested mode needs a capability the card does not declare (for example steering
    /// without [`HarnessCapability::Steer`](crate::HarnessCapability::Steer)), and
    /// [`HarnessError::State`](crate::HarnessError::State) when no cycle is running or the
    /// observation channel is closed.
    async fn send(&self, input: HarnessInput, mode: DeliveryMode) -> HarnessResult<SendReceipt>;

    /// Accepts input with [`DeliveryMode::Auto`] and returns its message-to-turn association.
    ///
    /// Convenience wrapper over [`send`](Self::send) matching Python's `mode=DeliveryMode.AUTO`
    /// keyword default; the call still does not wait for the turn to finish.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`send`](Self::send).
    async fn send_auto(&self, input: HarnessInput) -> HarnessResult<SendReceipt> {
        self.send(input, DeliveryMode::Auto).await
    }

    /// Aborts the active turn according to a declared capability.
    ///
    /// [`AbortMode::Graceful`] requires `GRACEFUL_ABORT` and lets the provider finish the current
    /// step before terminating the turn; [`AbortMode::Force`] requires `FORCE_ABORT` and stops it
    /// immediately.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::UnsupportedCapability`](crate::HarnessError::UnsupportedCapability)
    /// when the card does not declare
    /// [`HarnessCapability::GracefulAbort`](crate::HarnessCapability::GracefulAbort)
    /// (`AbortMode::Graceful`) or
    /// [`HarnessCapability::ForceAbort`](crate::HarnessCapability::ForceAbort)
    /// (`AbortMode::Force`), and [`HarnessError::State`](crate::HarnessError::State) when no turn
    /// is active.
    async fn abort(&self, mode: AbortMode) -> HarnessResult<()>;

    /// Aborts the active turn gracefully.
    ///
    /// Convenience wrapper over [`abort`](Self::abort) matching Python's
    /// `mode=AbortMode.GRACEFUL` keyword default.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`abort`](Self::abort) with [`AbortMode::Graceful`].
    async fn abort_graceful(&self) -> HarnessResult<()> {
        self.abort(AbortMode::Graceful).await
    }

    /// Pauses the active turn when `PAUSE_RESUME` is supported.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::UnsupportedCapability`](crate::HarnessError::UnsupportedCapability)
    /// when the card does not declare
    /// [`HarnessCapability::PauseResume`](crate::HarnessCapability::PauseResume), and
    /// [`HarnessError::State`](crate::HarnessError::State) when no turn is active or the harness is
    /// already paused.
    async fn pause(&self) -> HarnessResult<()>;

    /// Resumes a warm or checkpoint-restored paused turn.
    ///
    /// `query` optionally supplies the input that continues the paused turn; `None` resumes with
    /// the state captured at pause time.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::UnsupportedCapability`](crate::HarnessError::UnsupportedCapability)
    /// when the card does not declare
    /// [`HarnessCapability::PauseResume`](crate::HarnessCapability::PauseResume), and
    /// [`HarnessError::State`](crate::HarnessError::State) when no turn is paused.
    async fn resume(&self, query: Option<HarnessInput>) -> HarnessResult<()>;

    /// Returns the latest versioned provider snapshot for this agent.
    ///
    /// Returns `None` when the provider has nothing to persist yet. Implementations should also
    /// publish recoverable checkpoints through
    /// [`HarnessContext::checkpoint_sink`] whenever provider state changes materially, instead of
    /// relying on the host to poll this method.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::UnsupportedCapability`](crate::HarnessError::UnsupportedCapability)
    /// when the card does not declare
    /// [`HarnessCapability::Checkpoint`](crate::HarnessCapability::Checkpoint), and
    /// [`HarnessError::State`](crate::HarnessError::State) when the provider cannot serialize its
    /// current state.
    async fn export_checkpoint(&self) -> HarnessResult<Option<HarnessCheckpoint>>;
}

/// Optional model control a harness may implement next to [`HarnessProtocol`].
///
/// This is deliberately a separate trait rather than extra methods on [`HarnessProtocol`], so
/// existing implementors stay conformant: a harness that does not offer model control simply does
/// not implement this trait. Hosts check
/// [`card.supports(HarnessCapability::ModelDiscovery)`](HarnessCard::supports) /
/// [`card.supports(HarnessCapability::ModelSelection)`](HarnessCard::supports) before calling,
/// while an implementation that does not declare the capability must raise
/// [`HarnessError::UnsupportedCapability`](crate::HarnessError::UnsupportedCapability).
#[async_trait]
#[allow(clippy::needless_pass_by_value)] // DESIGN-fixed signature takes an owned selection.
pub trait HarnessModelControl: Send + Sync {
    /// Probes the models the provider currently offers.
    ///
    /// Works on a started harness (reads the live session) and on an unstarted one (performs a
    /// short handshake from provider configuration without issuing a model request). Requires
    /// [`HarnessCapability::ModelDiscovery`](crate::HarnessCapability::ModelDiscovery).
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::UnsupportedCapability`](crate::HarnessError::UnsupportedCapability)
    /// when the card does not declare
    /// [`HarnessCapability::ModelDiscovery`](crate::HarnessCapability::ModelDiscovery), and
    /// [`HarnessError::Protocol`](crate::HarnessError::Protocol) when the provider handshake fails.
    async fn list_models(&self) -> HarnessResult<Vec<ModelOption>>;

    /// Switches the model and/or reasoning effort used for the following turns.
    ///
    /// Requires [`HarnessCapability::ModelSelection`](crate::HarnessCapability::ModelSelection)
    /// and a started harness. The selection is applied immediately while no turn runs, otherwise
    /// before the next turn starts; a running turn keeps its model. The selection survives provider
    /// reconnects within the cycle.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::UnsupportedCapability`](crate::HarnessError::UnsupportedCapability)
    /// when the card does not declare
    /// [`HarnessCapability::ModelSelection`](crate::HarnessCapability::ModelSelection),
    /// [`HarnessError::State`](crate::HarnessError::State) when the harness has not been started,
    /// and [`HarnessError::Validation`](crate::HarnessError::Validation) when the provider cannot
    /// run the selected model or effort.
    async fn set_model(&self, selection: ModelSelection) -> HarnessResult<()>;
}

/// Factory SPI used to discover and construct third-party harnesses.
///
/// The trait is synchronous because a provider only constructs: the returned harness is
/// **unstarted** and becomes usable through a separate [`HarnessProtocol::start`] call, which is
/// where host compatibility and configuration are exercised against a live cycle.
#[allow(clippy::needless_pass_by_value)] // DESIGN-fixed signature takes the owned create mapping.
pub trait HarnessProvider: Send + Sync {
    /// Returns metadata for the harnesses created by this provider.
    #[must_use]
    fn card(&self) -> &HarnessCard;

    /// Validates provider configuration and creates an unstarted harness.
    ///
    /// `config` is the provider-defined create mapping handed to the harness verbatim; the provider
    /// owns its schema and must reject values it cannot run. The returned harness is **unstarted**:
    /// `create` only constructs, so [`HarnessProtocol::start`] is a separate call.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Validation`](crate::HarnessError::Validation) when `config` is not a
    /// valid configuration for this provider, and
    /// [`HarnessError::Protocol`](crate::HarnessError::Protocol) when construction fails.
    fn create(&self, config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>>;
}

/// Optional provider hook mapping a host's model choice onto provider configuration.
///
/// A [`HarnessProvider`] implements this next to [`HarnessProvider::create`] so a host never needs
/// to know the provider's configuration fields: the host picks a model (and effort) from the
/// provider's built-in catalog or from an endpoint it owns, and merges the returned fragment into
/// the `create` mapping.
///
/// # Capability gating
///
/// A choice carrying an [`endpoint`](ModelChoice::endpoint) requires the card to declare
/// [`HarnessCapability::EndpointModels`](crate::HarnessCapability::EndpointModels); `fallback` is
/// the model the provider may switch to on its own after an authentication failure and requires
/// [`HarnessCapability::AuthFallback`](crate::HarnessCapability::AuthFallback) — providers without
/// such a mechanism raise when a fallback is given.
pub trait ModelConfigProvider: Send + Sync {
    /// Returns the configuration fragment that runs `choice`.
    ///
    /// The fragment is merged into the [`HarnessProvider::create`] mapping by the host.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Validation`](crate::HarnessError::Validation) when the choice is
    /// invalid or the provider cannot run it (for example an effort its runtime has no selector
    /// for), and
    /// [`HarnessError::UnsupportedCapability`](crate::HarnessError::UnsupportedCapability) when the
    /// choice needs a capability the card does not declare.
    fn model_config(
        &self,
        choice: &ModelChoice,
        fallback: Option<&ModelChoice>,
    ) -> HarnessResult<JsonObject>;
}

/// Shared harness handle: a reference-counted [`HarnessProtocol`] implementation.
pub type DynHarness = Arc<dyn HarnessProtocol>;

/// Shared provider handle: a reference-counted [`HarnessProvider`] implementation.
pub type DynHarnessProvider = Arc<dyn HarnessProvider>;

/// Shared model-control handle: a reference-counted [`HarnessModelControl`] implementation.
pub type DynModelControl = Arc<dyn HarnessModelControl>;

#[cfg(test)]
mod tests {
    //! Object-safety smoke test: the SPI traits must be usable as shared trait objects.
    //! The full behavioral suite lives in `tests/protocol.rs`.

    use super::{DynHarness, DynHarnessProvider, DynModelControl, HarnessProtocol};

    #[test]
    fn traits_are_object_safe() {
        // Compile-time proof: these aliases only exist when each trait is object-safe with
        // `&self` receivers, and the boxed form is what `HarnessProvider::create` returns.
        let harness = std::any::type_name::<DynHarness>();
        assert!(harness.contains("dyn"), "{harness}");
        assert!(harness.contains("HarnessProtocol"), "{harness}");
        assert!(std::any::type_name::<DynHarnessProvider>().contains("dyn"));
        assert!(std::any::type_name::<DynModelControl>().contains("dyn"));
        assert!(std::any::type_name::<Box<dyn HarnessProtocol>>().contains("dyn"));
    }
}
