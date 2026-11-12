//! The compile-time built-in provider table.
//!
//! [`builtin_provider`] is the resolution half of the registry: it maps a registry name onto the
//! provider that implements it, or reports why this build cannot serve the name. Unknown names
//! return [`None`] so the factory can fall back to in-process registrations; every declared name
//! returns a result, so a known-but-deferred provider produces an `Unsupported` error instead of a
//! "not found".
//!
//! Feature gating is expressed here, not in the provider modules: without the `native` feature the
//! native entry point is compiled out, and without `vendor` the three vendor stubs stay
//! unregistered. The module itself is always declared because the factory consults it in every
//! feature combination.
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};

use crate::provider::{provider_names, ProviderSpec};
// Only the implemented providers register a model-configuration hook; a build with every provider
// compiled out has nothing to wrap.
#[cfg(any(feature = "native", feature = "vendor"))]
use crate::provider::model_configuring_provider;

#[cfg(feature = "native")]
pub mod native;
#[cfg(feature = "vendor")]
pub mod vendor;

#[cfg(feature = "vendor")]
pub mod claudecode;
#[cfg(feature = "vendor")]
pub mod codex;
#[cfg(feature = "vendor")]
pub mod dsh;

/// Message of a `native_v2` resolution: the provider lives outside this crate.
const NATIVE_V2_DEFERRED: &str =
    "harness provider 'native_v2' lives in openjiuwen-agent-teams and is not wired into this build";

/// Resolves a built-in registry name to its provider.
///
/// Returns [`None`] for a name this crate does not declare, and the provider (or the reason this
/// build cannot serve it) for every declared name.
#[must_use]
pub fn builtin_provider(name: &str) -> Option<Result<Arc<dyn ProviderSpec>>> {
    match name {
        provider_names::NATIVE => Some(native_provider()),
        provider_names::NATIVE_V2 => Some(Err(Error::unsupported(NATIVE_V2_DEFERRED))),
        provider_names::CLAUDE_CODE | provider_names::CODEX | provider_names::DSH => {
            Some(vendor_provider(name))
        }
        _ => None,
    }
}

/// The native provider, or the reason this build has it compiled out.
// Both feature variants must keep the same signature so `builtin_provider` stays feature-agnostic;
// with the `native` feature the error arm simply never fires.
#[cfg(feature = "native")]
#[allow(clippy::unnecessary_wraps)]
fn native_provider() -> Result<Arc<dyn ProviderSpec>> {
    Ok(model_configuring_provider(native::provider()))
}

/// The native provider, or the reason this build has it compiled out.
#[cfg(not(feature = "native"))]
fn native_provider() -> Result<Arc<dyn ProviderSpec>> {
    Err(Error::unsupported(
        "harness provider 'native' is disabled; rebuild with the `native` feature",
    ))
}

/// The vendor stub for `name`, or the reason this build has it compiled out.
#[cfg(feature = "vendor")]
fn vendor_provider(name: &str) -> Result<Arc<dyn ProviderSpec>> {
    match name {
        provider_names::CLAUDE_CODE => Ok(model_configuring_provider(claudecode::provider())),
        provider_names::CODEX => Ok(model_configuring_provider(codex::provider())),
        provider_names::DSH => Ok(model_configuring_provider(dsh::provider())),
        // `builtin_provider` routes only the three names above; anything else is a caller error.
        other => Err(Error::validation(format!(
            "'{other}' is not a vendor harness provider"
        ))),
    }
}

/// The vendor stub for `name`, or the reason this build has it compiled out.
#[cfg(not(feature = "vendor"))]
fn vendor_provider(name: &str) -> Result<Arc<dyn ProviderSpec>> {
    Err(Error::unsupported(format!(
        "vendor harness provider '{name}' is deferred; rebuild with the `vendor` feature to \
         register the stub"
    )))
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use openjiuwen_common::error::Error;

    use super::builtin_provider;
    use crate::provider::provider_names;

    #[test]
    fn unknown_names_are_not_builtin() {
        assert!(builtin_provider("openjiuwen").is_none());
        assert!(builtin_provider("").is_none());
    }

    #[test]
    fn native_v2_reports_the_sibling_crate() {
        let error = builtin_provider(provider_names::NATIVE_V2)
            .expect("native_v2 is a declared name")
            .err()
            .expect("native_v2 is not wired into this build");
        assert!(matches!(error, Error::Unsupported(_)));
        assert!(
            error.to_string().contains("openjiuwen-agent-teams"),
            "unexpected message: {error}"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn native_resolves_to_the_harness_provider() {
        let spec = builtin_provider(provider_names::NATIVE)
            .expect("native is a declared name")
            .expect("the native feature is enabled");
        assert_eq!(spec.card().name, openjiuwen_harness::protocol::HARNESS_NAME);
        assert!(spec.configures_models());
    }

    #[cfg(not(feature = "native"))]
    #[test]
    fn native_reports_the_missing_feature() {
        let error = builtin_provider(provider_names::NATIVE)
            .expect("native is a declared name")
            .err()
            .expect("the native feature is disabled");
        assert!(matches!(error, Error::Unsupported(_)));
        assert!(
            error
                .to_string()
                .contains("rebuild with the `native` feature"),
            "unexpected message: {error}"
        );
    }

    #[cfg(not(feature = "vendor"))]
    #[test]
    fn vendor_names_report_the_missing_feature() {
        for name in [
            provider_names::CLAUDE_CODE,
            provider_names::CODEX,
            provider_names::DSH,
        ] {
            let error = builtin_provider(name)
                .expect("vendor names are declared")
                .err()
                .expect("the vendor feature is disabled");
            assert!(matches!(error, Error::Unsupported(_)));
            assert!(
                error.to_string().contains(name)
                    && error
                        .to_string()
                        .contains("rebuild with the `vendor` feature"),
                "unexpected message for {name}: {error}"
            );
        }
    }

    #[cfg(feature = "vendor")]
    #[test]
    fn vendor_names_resolve_when_enabled() {
        for (name, card_name) in [
            (provider_names::CLAUDE_CODE, "claude-code"),
            (provider_names::CODEX, "codex"),
            (provider_names::DSH, "deepseek-harness"),
        ] {
            let spec = builtin_provider(name)
                .expect("vendor names are declared")
                .expect("the vendor feature is enabled");
            assert_eq!(spec.card().name, card_name);
        }
    }
}
