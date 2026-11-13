//! Real remote sandbox providers for the `aio`, `jiuwenbox`, and `yuanrong`
//! sandbox types.
//!
//! Python parity: `openjiuwen/extensions/sys_operation/sandbox/providers/`
//! (v0.1.20). Those three modules each implement the kernel's filesystem, shell,
//! and code operations by talking to a container, and this subtree does the
//! same instead of registering the deferred sets of
//! [`crate::sys_operation::register_deferred`]. The parents in
//! [`crate::sys_operation`] stay the registration facade.
//!
//! # Transports
//!
//! - `aio` speaks the kernel gateway envelope
//!   ([`GatewayInvokeRequest`](openjiuwen_core::sys_operation::sandbox::config::GatewayInvokeRequest) /
//!   [`GatewayResponse`](openjiuwen_core::sys_operation::sandbox::config::GatewayResponse))
//!   over HTTP to an AIO container.
//! - `jiuwenbox` speaks the container's native REST API under `/api/v1/...`.
//! - `yuanrong` speaks the ADX API under `/api/environment/v1/...` plus
//!   `/direct/{id}/invoke`.
//!
//! # Layering
//!
//! [`transport`] owns the HTTP client, the auth injection, the error type, and
//! the `OpResult` mapping; [`semantics`] owns the pure host-side behavior every
//! provider shares (line selection, newline transforms, chunking, path and
//! quoting helpers, list sorting/filtering); [`local_io`] owns the host-side
//! file reads and writes of transfers; `backend` owns the provider-neutral
//! `SandboxBackend` primitives and the generic wrappers that implement the
//! kernel operation traits once. A provider module then only translates its own
//! wire formats and optionally overrides `code_command` / `code_cwd`.
//!
//! # Lifecycle
//!
//! The kernel has no launcher or lifecycle API: it resolves a
//! [`SandboxEndpoint`] from configuration and calls the factories. A provider
//! therefore creates (and caches) its container lazily, keyed by
//! `base_url|isolation_key` as the
//! [`SandboxProviderRegistry`](openjiuwen_core::sys_operation::sandbox::provider::SandboxProviderRegistry)
//! seam requires, because a provider cannot feed a discovered sandbox id back
//! into the endpoint.
//!
//! # Errors
//!
//! Provider failures become the kernel's `SandboxGatewayError` result (code
//! `199007`) through
//! [`sandbox_gateway_message`](openjiuwen_core::sys_operation::status::sandbox_gateway_message),
//! or a plain [`openjiuwen_common::error::Error`] where an operation reports
//! through `Result`. Credentials, auth query parameters, and payload contents
//! are never logged.

pub(crate) mod semantics;

mod aio;
mod backend;
mod jiuwenbox;
mod local_io;
mod transport;
mod yuanrong;

pub(crate) use backend::{
    provider_set_from, ExecCommand, ExecRequest, ExecResult, ListRequest, RemoteCode, RemoteEntry,
    RemoteFs, RemoteShell, SandboxBackend, SearchRequest,
};
pub(crate) use transport::{gateway_failure, remote_error, RawResponse, RemoteError, SandboxHttp};

use openjiuwen_common::error::{Error, Result};

use crate::sys_operation::SandboxProviderSet;

/// The three sandbox types Python registers from
/// `extensions/sys_operation/sandbox/providers/`.
pub(crate) const SUPPORTED_SANDBOX_TYPES: [&str; 3] = ["aio", "jiuwenbox", "yuanrong"];

/// The full provider set for one supported remote sandbox type.
///
/// # Errors
///
/// Returns [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
/// for a sandbox type this subtree does not implement.
pub(crate) fn provider_set(sandbox_type: &str) -> Result<SandboxProviderSet> {
    match sandbox_type {
        "aio" => Ok(aio::provider_set()),
        "jiuwenbox" => Ok(jiuwenbox::provider_set()),
        "yuanrong" => Ok(yuanrong::provider_set()),
        other => Err(Error::unsupported(format!(
            "sandbox provider '{other}' is not a remote provider; supported types: aio, jiuwenbox, yuanrong"
        ))),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn every_supported_type_has_a_complete_provider_set() {
        for sandbox_type in SUPPORTED_SANDBOX_TYPES {
            let set = provider_set(sandbox_type).unwrap();
            assert!(!set.is_empty(), "{sandbox_type} must register a factory");
            assert!(
                set.fs.is_some(),
                "{sandbox_type} needs a filesystem factory"
            );
            assert!(set.shell.is_some(), "{sandbox_type} needs a shell factory");
            assert!(set.code.is_some(), "{sandbox_type} needs a code factory");
        }
    }

    #[test]
    fn an_unknown_type_is_unsupported() {
        let error = provider_set("local").err().unwrap();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.message(),
            "sandbox provider 'local' is not a remote provider; supported types: aio, jiuwenbox, yuanrong"
        );
    }
}
