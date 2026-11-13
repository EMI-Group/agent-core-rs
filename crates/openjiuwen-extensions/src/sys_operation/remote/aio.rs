//! The `aio` remote sandbox backend.
//!
//! Python parity: `extensions/sys_operation/sandbox/providers/aio.py`, which
//! drives an AIO container through the kernel gateway envelope over HTTP.
//!
//! Placeholder: the real transport replaces this file; until then the three
//! factories answer `Unsupported`.

use crate::sys_operation::{SandboxEndpoint, SandboxGatewayConfig, SandboxProviderSet};
use openjiuwen_common::error::Error;
use std::sync::Arc;

/// The filesystem, shell, and code factories of the `aio` sandbox type.
pub(crate) fn provider_set() -> SandboxProviderSet {
    SandboxProviderSet::new()
        .with_fs(Arc::new(
            |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                Err(Error::unsupported("aio provider is not implemented yet"))
            },
        ))
        .with_shell(Arc::new(
            |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                Err(Error::unsupported("aio provider is not implemented yet"))
            },
        ))
        .with_code(Arc::new(
            |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                Err(Error::unsupported("aio provider is not implemented yet"))
            },
        ))
}
