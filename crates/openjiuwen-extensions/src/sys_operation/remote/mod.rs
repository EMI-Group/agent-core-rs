//! Real remote sandbox providers for the `aio`, `jiuwenbox`, and `yuanrong`
//! sandbox types.
//!
//! Python parity: `openjiuwen/extensions/sys_operation/sandbox/providers/`
//! (v0.1.20). Those three modules each implement the kernel's filesystem, shell,
//! and code operations by talking to a container, and this subtree does the
//! same instead of leaving the three types without a transport.
//! [`crate::sys_operation::register_deferred`] installs all three, and the
//! parents in [`crate::sys_operation`] stay the registration facade.
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
//! `transport` owns the HTTP client, the auth injection, the error type, and
//! the `OpResult` mapping; `semantics` owns the pure host-side behavior every
//! provider shares (line selection, newline transforms, chunking, path and
//! quoting helpers, list sorting/filtering); `local_io` owns the host-side
//! file reads and writes of transfers.
//!
//! `backend/` owns the provider-neutral `SandboxBackend` primitives and the
//! generic wrappers that implement the kernel operation traits once: `mod.rs`
//! holds the request/result types, the trait, and the factory constructor,
//! while `backend/fs.rs`, `backend/shell.rs`, and `backend/code.rs` hold the
//! filesystem, shell, and code wrappers. A provider module then only translates
//! its own wire formats and optionally overrides `code_command` / `code_cwd`.
//!
//! Each provider is its own subtree: `aio/` holds its shared envelope client in
//! `mod.rs` and the three wrappers in `aio/fs.rs`, `aio/shell.rs`, and
//! `aio/code.rs`; `jiuwenbox.rs` and `yuanrong.rs` stay single files built on
//! the generic backend wrappers. Every provider keeps its tests in a sibling
//! `tests.rs` (with per-operation child modules where the suite is large).
//!
//! # Lifecycle
//!
//! The kernel has no launcher or lifecycle API: it resolves a
//! [`SandboxEndpoint`](crate::sys_operation::SandboxEndpoint) from configuration
//! and calls the factories. A provider therefore creates (and caches) its
//! container lazily, keyed by `base_url|isolation_key` as the
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

mod aio;
mod backend;
mod conditional;
mod jiuwenbox;
mod local_io;
mod semantics;
mod teardown;
mod transport;
mod yuanrong;

use openjiuwen_common::error::{Error, Result};

pub use self::teardown::{SandboxSelector, TeardownFailure, TeardownReport};
use crate::sys_operation::SandboxProviderSet;

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

/// One process-wide guard shared by every teardown suite in this subtree.
///
/// [`SandboxSelector::all`] drains the whole process cache, so a drain in one
/// provider suite can delete the container another teardown test is about to
/// release. Every teardown test therefore serialises on this single lock
/// rather than a per-module one.
#[cfg(test)]
pub(crate) fn teardown_guard() -> &'static tokio::sync::Mutex<()> {
    static GUARD: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    &GUARD
}
/// Releases the cached containers of one remote sandbox type.
///
/// `aio` has no container-delete API (the gateway reaps idle containers), so it
/// answers [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
/// naming the gateway; `jiuwenbox` and `yuanrong` delete their cached
/// containers and tolerate an already-gone container.
///
/// # Errors
///
/// Returns [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
/// for the `aio` sandbox type and for an unknown type, and
/// [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
/// for a selector that names nothing to release.
pub async fn teardown_remote(
    sandbox_type: &str,
    selector: &SandboxSelector,
) -> Result<TeardownReport> {
    match sandbox_type {
        "aio" => Err(teardown::aio_teardown_error()),
        "jiuwenbox" => jiuwenbox::teardown(selector).await,
        "yuanrong" => yuanrong::teardown(selector).await,
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
        for sandbox_type in crate::sys_operation::REMOTE_SANDBOX_TYPES {
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

    #[tokio::test]
    async fn teardown_rejects_aio_and_unknown_types() {
        let error = teardown_remote("aio", &SandboxSelector::all())
            .await
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.message().contains("gateway"), "{}", error.message());

        let error = teardown_remote("local", &SandboxSelector::all())
            .await
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.message(),
            "sandbox provider 'local' is not a remote provider; supported types: aio, jiuwenbox, yuanrong"
        );
    }

    #[test]
    fn a_scope_key_matches_the_selector_key() {
        assert_eq!(
            teardown::shared_key("127.0.0.1:9/", Some("agent")),
            "http://127.0.0.1:9|agent"
        );
        assert_eq!(
            SandboxSelector::scope("127.0.0.1:9/", Some("agent"))
                .shared_key
                .as_deref(),
            Some("http://127.0.0.1:9|agent")
        );
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
