//! The `yuanrong` remote sandbox backend.
//!
//! Python parity: `extensions/sys_operation/sandbox/providers/yuanrong.py`,
//! which drives a `YuanRong` ADX environment through `/api/environment/v1/...`
//! and `/direct/{id}/invoke`.
//!
//! Placeholder: the real transport replaces this file; until then the three
//! factories answer `Unsupported`.

use crate::sys_operation::{SandboxEndpoint, SandboxGatewayConfig, SandboxProviderSet};
use openjiuwen_common::error::Error;
use std::sync::Arc;

/// The filesystem, shell, and code factories of the `yuanrong` sandbox type.
pub(crate) fn provider_set() -> SandboxProviderSet {
    SandboxProviderSet::new()
        .with_fs(Arc::new(
            |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                Err(Error::unsupported(
                    "yuanrong provider is not implemented yet",
                ))
            },
        ))
        .with_shell(Arc::new(
            |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                Err(Error::unsupported(
                    "yuanrong provider is not implemented yet",
                ))
            },
        ))
        .with_code(Arc::new(
            |_endpoint: &SandboxEndpoint, _config: &SandboxGatewayConfig| {
                Err(Error::unsupported(
                    "yuanrong provider is not implemented yet",
                ))
            },
        ))
}
