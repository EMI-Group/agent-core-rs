//! Sandbox backend: gateway configuration, the provider and launcher seams, container
//! lifecycle, and the remote client.
//!
//! The configuration types always compile because they are part of a card's serialized
//! shape. Two extension seams sit above them: [`provider`] builds the filesystem/shell/code
//! backends for a resolved endpoint, and [`launcher`] creates, suspends, resumes, and
//! destroys the container that endpoint names. [`lifecycle`] keeps one record per isolation
//! key so a container is reused (and idle ones are stopped) instead of relaunched.
//! `openjiuwen-extensions` implements the providers; a host supplies its container
//! launchers.
//!
//! The HTTP gateway client in `http` is gated behind the `sandbox` feature and reports
//! [`openjiuwen_common::error::Error::Unsupported`] for remote calls, because the AIO
//! container protocol is an external service contract this rewrite does not implement.
//!
//! ```
//! use openjiuwen_sysop::sandbox::config::{ContainerScope, SandboxGatewayConfig};
//!
//! let config = SandboxGatewayConfig::default();
//! assert_eq!(config.isolation.container_scope, ContainerScope::Session);
//! ```
pub mod config;
pub mod launcher;
pub mod lifecycle;
pub mod operation;
pub mod provider;

#[cfg(feature = "sandbox")]
pub mod http;
