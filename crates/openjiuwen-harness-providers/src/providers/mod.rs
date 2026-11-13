//! The compile-time built-in provider table.
//!
//! [`builtin_provider`] is the resolution half of the registry: it maps a registry name onto the
//! provider that implements it, or reports why this build cannot serve the name. Unknown names
//! return [`None`] so the factory can fall back to in-process registrations; every declared name
//! returns a result, so a known-but-deferred provider produces an `Unsupported` error instead of a
//! "not found".
//!
//! Feature gating is expressed here, not in the provider modules: without the `native` feature the
//! native entry point is compiled out, without `vendor` the three vendor stubs stay unregistered,
//! and without `native-v2` the agent-teams-hosted provider is absent. The module itself is always
//! declared because the factory consults it in every feature combination.
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

/// The `native_v2` provider: an in-process harness hosted by `openjiuwen-agent-teams`.
#[cfg(feature = "native-v2")]
pub mod native_v2;

/// Message of a `native_v2` resolution when this build has the feature compiled out.
#[cfg(not(feature = "native-v2"))]
const NATIVE_V2_DEFERRED: &str =
    "harness provider 'native_v2' is disabled; rebuild with the `native-v2` feature to register \
     the openjiuwen-agent-teams provider";

/// Resolves a built-in registry name to its provider.
///
/// Returns [`None`] for a name this crate does not declare, and the provider (or the reason this
/// build cannot serve it) for every declared name.
#[must_use]
pub fn builtin_provider(name: &str) -> Option<Result<Arc<dyn ProviderSpec>>> {
    match name {
        provider_names::NATIVE => Some(native_provider()),
        provider_names::NATIVE_V2 => Some(native_v2_provider()),
        provider_names::CLAUDE_CODE | provider_names::CODEX | provider_names::DSH => {
            Some(vendor_provider(name))
        }
        _ => None,
    }
}

/// The `native_v2` provider, or the reason this build has it compiled out.
///
/// The provider has no model-configuration hook, so it registers through
/// [`plain_provider`](crate::provider::plain_provider), not the model-configuring wrapper.
#[cfg(feature = "native-v2")]
#[allow(clippy::unnecessary_wraps)]
fn native_v2_provider() -> Result<Arc<dyn ProviderSpec>> {
    Ok(crate::provider::plain_provider(native_v2::provider()))
}

/// The `native_v2` provider, or the reason this build has it compiled out.
#[cfg(not(feature = "native-v2"))]
fn native_v2_provider() -> Result<Arc<dyn ProviderSpec>> {
    Err(Error::unsupported(NATIVE_V2_DEFERRED))
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

    #[cfg(not(feature = "native-v2"))]
    #[test]
    fn native_v2_reports_the_missing_feature() {
        let error = builtin_provider(provider_names::NATIVE_V2)
            .expect("native_v2 is a declared name")
            .err()
            .expect("the native-v2 feature is disabled");
        assert!(matches!(error, Error::Unsupported(_)));
        assert!(
            error
                .to_string()
                .contains("rebuild with the `native-v2` feature"),
            "unexpected message: {error}"
        );
    }

    #[cfg(feature = "native-v2")]
    #[test]
    fn native_v2_resolves_to_the_team_hosted_provider() {
        let spec = builtin_provider(provider_names::NATIVE_V2)
            .expect("native_v2 is a declared name")
            .expect("the native-v2 feature is enabled");
        assert_eq!(spec.card().name, "native_v2");
        assert_eq!(spec.card().implementation_version, "0.2.0");
        assert!(
            !spec.configures_models(),
            "native_v2 configures no models: members get theirs from the harness factory"
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
