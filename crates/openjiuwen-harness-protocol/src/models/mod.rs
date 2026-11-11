//! Harness data models: identity and capability negotiation, input delivery, and model control.
//!
//! Submodules:
//!
//! - [`card`] — `PROTOCOL_VERSION`, [`HarnessCapability`], [`HostCapability`], and
//!   [`HarnessCard`] with its `validate_host` negotiation check.
//! - [`delivery`] — [`DeliveryMode`], [`AbortMode`], [`HarnessInput`], and [`SendReceipt`].
//! - [`model_control`] — [`ModelSelection`], [`ModelEndpoint`], [`ModelChoice`],
//!   [`ModelOption`], and the auth-fallback request/event type constants.
//!
//! Everything is re-exported from the crate root as well, mirroring Python's eager
//! `__init__.__all__`.

pub mod card;
pub mod context;
pub mod delivery;
pub mod model_control;

pub use card::{
    default_compatible_protocol_versions, HarnessCapability, HarnessCard, HostCapability,
    PROTOCOL_VERSION,
};
pub use delivery::{AbortMode, DeliveryMode, HarnessInput, SendReceipt};
pub use model_control::{
    ModelChoice, ModelEndpoint, ModelOption, ModelSelection, AUTH_FALLBACK_EVENT_TYPE,
    AUTH_FALLBACK_REQUEST_TYPE,
};
