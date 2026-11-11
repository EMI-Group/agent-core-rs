//! Sandbox backend: gateway configuration, the provider seam, and the remote client.
//!
//! The configuration types always compile because they are part of a card's serialized
//! shape. The provider traits in [`provider`] are the extension seam
//! (`openjiuwen-extensions` implements them); the HTTP gateway client in [`http`] is
//! gated behind the `sandbox` feature and reports [`openjiuwen_common::error::Error::Unsupported`]
//! for remote calls, because the AIO container protocol is an external service contract
//! this rewrite does not implement.
//!
//! ```
//! use openjiuwen_sysop::sandbox::config::{ContainerScope, SandboxGatewayConfig};
//!
//! let config = SandboxGatewayConfig::default();
//! assert_eq!(config.isolation.container_scope, ContainerScope::Session);
//! ```

pub mod config;
pub mod operation;
pub mod provider;

#[cfg(feature = "sandbox")]
pub mod http;
