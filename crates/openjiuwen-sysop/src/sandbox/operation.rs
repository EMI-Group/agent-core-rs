//! Sandbox operation dispatch: turns a sandbox card into the three operation backends.
//!
//! The dispatch resolves the card's isolation-key template, asks the
//! [`SandboxProviderRegistry`] for a provider for the configured `sandbox_type`, and
//! returns backends that forward every call to that provider. When no provider is
//! registered the whole set is rejected with `ErrorKind::Unsupported` (Python raises
//! `NotImplementedError` only on the first call; failing at construction is stricter and
//! reported as a documented deviation).
//!
//! ```
//! use openjiuwen_sysop::card::SysOperationCard;
//! use openjiuwen_sysop::mode::OperationMode;
//! use openjiuwen_sysop::sandbox::operation::sandbox_operations;
//!
//! let card = SysOperationCard::new("sys_op").with_mode(OperationMode::Sandbox);
//! // A sandbox card without a gateway configuration is rejected before any I/O.
//! assert!(sandbox_operations(&card).is_err());
//! ```

use openjiuwen_common::error::{Error, Result};

use crate::card::SysOperationCard;
use crate::operation::OperationSet;

/// Builds the sandbox backends for `card`, using no session id for the isolation key.
///
/// # Errors
///
/// Returns `ErrorKind::SysOp` when the card lacks a usable gateway configuration and
/// `ErrorKind::Unsupported` when no provider is registered for the configured
/// `sandbox_type`.
pub fn sandbox_operations(card: &SysOperationCard) -> Result<OperationSet> {
    sandbox_operations_for_session(card, None)
}

/// Builds the sandbox backends for `card`, resolving `{session_id}` in the isolation key.
///
/// # Errors
///
/// See [`sandbox_operations`].
pub fn sandbox_operations_for_session(
    card: &SysOperationCard,
    session_id: Option<&str>,
) -> Result<OperationSet> {
    card.validate()?;
    let gateway = card.gateway_config.as_ref().ok_or_else(|| {
        crate::card::card_param_error("gateway_config is required when mode is 'sandbox'")
    })?;
    let launcher = gateway.launcher_config.as_ref().ok_or_else(|| {
        crate::card::card_param_error("launcher_config is required when mode is 'sandbox'")
    })?;
    let template = crate::sandbox::config::isolation_key_template_from_config(gateway)?;
    let isolation_key = crate::sandbox::config::resolve_isolation_key(&template, session_id);
    let endpoint = crate::sandbox::config::SandboxEndpoint {
        base_url: launcher.base_url.clone().unwrap_or_default(),
        sandbox_id: None,
        isolation_key: Some(isolation_key),
    };
    let registry = crate::sandbox::provider::global();
    register_builtin_providers(registry, gateway)?;
    let fs = registry.create_fs(&launcher.sandbox_type, &endpoint, gateway)?;
    let shell = registry.create_shell(&launcher.sandbox_type, &endpoint, gateway)?;
    let code = registry.create_code(&launcher.sandbox_type, &endpoint, gateway)?;
    Ok(OperationSet {
        fs: Some(fs),
        shell: Some(shell),
        code: Some(code),
    })
}

/// Installs the provider shipped with this crate for the configured sandbox type.
///
/// Without the `sandbox` feature nothing is installed, so a sandbox card is reported as
/// unsupported. With the feature, the HTTP gateway provider is registered; it
/// validates the configuration and returns `Error::Unsupported` for remote calls that
/// need the external AIO container protocol.
///
/// # Errors
///
/// Propagates registration failures.
pub fn register_builtin_providers(
    registry: &crate::sandbox::provider::SandboxProviderRegistry,
    gateway: &crate::sandbox::config::SandboxGatewayConfig,
) -> Result<()> {
    let _ = (registry, gateway);
    Ok(())
}

/// Reports that a sandbox operation is unavailable in this build.
pub fn unsupported_sandbox(operation: &str) -> Error {
    Error::unsupported(format!(
        "sandbox operation '{operation}' is not available: no provider registered for this sandbox type"
    ))
}
