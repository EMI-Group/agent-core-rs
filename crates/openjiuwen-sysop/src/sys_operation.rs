//! The `SysOperation` facade: one card, three operation backends, one mode.
//!
//! Python parity: `SysOperation` (`core/sys_operation/sys_operation.py`) is a factory plus
//! cache over the operation registry; `fs()`, `shell()`, and `code()` return the backend
//! selected by [`crate::card::SysOperationCard::mode`], and `None` when the mode does not
//! provide one.
//!
//! ```
//! use openjiuwen_sysop::card::SysOperationCard;
//! use openjiuwen_sysop::mode::{OperationKind, OperationMode};
//!
//! let card = SysOperationCard::new("sys_op").with_mode(OperationMode::Local);
//! assert!(card.validate().is_ok());
//! assert_eq!(
//!     card.tool_id(OperationKind::Fs, "read_file"),
//!     "sys_op.fs.read_file"
//! );
//! ```

use std::sync::Arc;

use openjiuwen_common::error::Result;

use crate::card::SysOperationCard;
use crate::code::types::CodeExecParams;
use crate::fs::types::ReadFileParams;
use crate::mode::{OperationKind, OperationMode};
use crate::operation::{CodeOperation, FsOperation, OperationSet, ShellOperation};
use crate::sandbox::config::{OnStop, SandboxEndpoint};
use crate::sandbox::lifecycle::SandboxLifecycle;
use crate::tool::{fs_tool_specs, shell_tool_specs, OperationToolSpec};

/// The container a sandbox-mode service was built against.
///
/// Recorded only by [`SysOperation::sandbox_with_lifecycle`], so [`SysOperation::release_sandbox`]
/// and [`SysOperation::evict_idle_sandboxes`] act on the lifecycle (and isolation key) the
/// endpoint was resolved from. A static pre-deployed sandbox has no binding.
#[derive(Clone)]
struct SandboxBinding {
    lifecycle: SandboxLifecycle,
    isolation_key: String,
    endpoint: SandboxEndpoint,
}

impl std::fmt::Debug for SandboxBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxBinding")
            .field("isolation_key", &self.isolation_key)
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

/// A configured sys-operation service.
#[derive(Clone)]
pub struct SysOperation {
    card: SysOperationCard,
    mode: OperationMode,
    operations: OperationSet,
    isolation_key_template: Option<String>,
    sandbox: Option<SandboxBinding>,
}

impl std::fmt::Debug for SysOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SysOperation")
            .field("id", &self.card.id)
            .field("mode", &self.mode)
            .field("operations", &self.operations)
            .field("isolation_key_template", &self.isolation_key_template)
            .field("sandbox", &self.sandbox)
            .finish()
    }
}

impl SysOperation {
    /// Builds the backends selected by `card.mode`.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::SysOp` carrying the Python card-parameter message (code 199002)
    /// when the card is inconsistent, and `ErrorKind::Unsupported` when the card selects a
    /// sandbox backend this build cannot provide (for example a sandbox card without the
    /// `sandbox` feature or without a registered provider).
    pub fn new(card: SysOperationCard) -> Result<Self> {
        card.validate()?;
        let mode = card.mode;
        let operations = match mode {
            OperationMode::Local => {
                let config = card.resolved_work_config();
                OperationSet {
                    fs: Some(crate::fs::local_fs(config.clone())?),
                    shell: Some(crate::shell::local_shell(config.clone())?),
                    code: Some(crate::code::local_code(config)?),
                }
            }
            OperationMode::Sandbox => crate::sandbox::operation::sandbox_operations(&card)?,
        };
        let isolation_key_template = if mode.is_sandbox() {
            Some(crate::sandbox::config::isolation_key_template_from_config(
                card.gateway_config
                    .as_ref()
                    .ok_or_else(|| crate::card::card_param_error("gateway_config is required"))?,
            )?)
        } else {
            None
        };
        Ok(Self {
            card,
            mode,
            operations,
            isolation_key_template,
            sandbox: None,
        })
    }

    /// Builds a local-mode service with the given work configuration.
    ///
    /// # Errors
    ///
    /// Propagates [`LocalWorkConfig::validate`](crate::config::LocalWorkConfig::validate).
    pub fn local(config: crate::config::LocalWorkConfig) -> Result<Self> {
        Self::new(SysOperationCard::new("sys_op").with_work_config(config))
    }

    /// Builds a sandbox-mode service with the given gateway configuration.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::SysOp` (code 199002) when the configuration is incomplete and
    /// `ErrorKind::Unsupported` when no sandbox provider is available.
    pub fn sandbox(
        sandbox_type: impl Into<String>,
        gateway_config: crate::sandbox::config::SandboxGatewayConfig,
    ) -> Result<Self> {
        let mut gateway_config = gateway_config;
        if let Some(launcher) = gateway_config.launcher_config.as_mut() {
            launcher.sandbox_type = sandbox_type.into();
        }
        Self::new(
            SysOperationCard::new("sys_op")
                .with_mode(OperationMode::Sandbox)
                .with_gateway_config(gateway_config),
        )
    }

    /// Builds a sandbox-mode service whose container is resolved through `lifecycle`.
    ///
    /// The endpoint is resolved once, before the provider backends are created, so a
    /// launched container's `base_url`/`sandbox_id` reach the providers (and are reported by
    /// [`Self::sandbox_endpoint`]). `session_id` selects the session a session-scoped
    /// isolation template targets; `now_epoch_secs` makes idle-stop deterministic.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::SysOp` (199002) when the card is inconsistent, (199008) when the
    /// launcher fails or (199010) when the isolation key is unusable, and
    /// `ErrorKind::Unsupported` when the launcher type or the sandbox provider is not
    /// registered.
    pub async fn sandbox_with_lifecycle(
        sandbox_type: impl Into<String>,
        gateway_config: crate::sandbox::config::SandboxGatewayConfig,
        session_id: Option<&str>,
        lifecycle: &SandboxLifecycle,
        now_epoch_secs: u64,
    ) -> Result<Self> {
        let mut gateway_config = gateway_config;
        if let Some(launcher) = gateway_config.launcher_config.as_mut() {
            launcher.sandbox_type = sandbox_type.into();
        }
        let card = SysOperationCard::new("sys_op")
            .with_mode(OperationMode::Sandbox)
            .with_gateway_config(gateway_config);
        let endpoint = crate::sandbox::operation::resolve_sandbox_endpoint(
            &card,
            session_id,
            lifecycle,
            now_epoch_secs,
        )
        .await?;
        let operations = crate::sandbox::operation::operations_for_endpoint(
            &card,
            endpoint.clone(),
            crate::sandbox::provider::global(),
        )?;
        let gateway = card.gateway_config.as_ref().ok_or_else(|| {
            crate::card::card_param_error("gateway_config is required when mode is 'sandbox'")
        })?;
        let isolation_key_template =
            crate::sandbox::config::isolation_key_template_from_config(gateway)?;
        Ok(Self {
            card,
            mode: OperationMode::Sandbox,
            operations,
            isolation_key_template: Some(isolation_key_template),
            sandbox: Some(SandboxBinding {
                lifecycle: lifecycle.clone(),
                isolation_key: endpoint.isolation_key.clone().unwrap_or_default(),
                endpoint,
            }),
        })
    }

    /// The card this service was built from.
    #[must_use]
    pub fn card(&self) -> &SysOperationCard {
        &self.card
    }
    /// The configured operation mode.
    #[must_use]
    pub const fn mode(&self) -> OperationMode {
        self.mode
    }

    /// The filesystem backend, if this mode provides one.
    #[must_use]
    pub fn fs(&self) -> Option<Arc<dyn FsOperation>> {
        self.operations.fs.clone()
    }

    /// The shell backend, if this mode provides one.
    #[must_use]
    pub fn shell(&self) -> Option<Arc<dyn ShellOperation>> {
        self.operations.shell.clone()
    }

    /// The code backend, if this mode provides one.
    #[must_use]
    pub fn code(&self) -> Option<Arc<dyn CodeOperation>> {
        self.operations.code.clone()
    }

    /// The sandbox isolation-key template, `None` in local mode
    /// (Python `SysOperation.isolation_key_template`).
    #[must_use]
    pub fn isolation_key_template(&self) -> Option<&str> {
        self.isolation_key_template.as_deref()
    }

    /// The container endpoint this service's sandbox backends target.
    ///
    /// `None` in local mode and for a static pre-deployed sandbox, which carries no
    /// lifecycle bookkeeping; [`Self::sandbox_with_lifecycle`] records it.
    #[must_use]
    pub fn sandbox_endpoint(&self) -> Option<&SandboxEndpoint> {
        self.sandbox.as_ref().map(|binding| &binding.endpoint)
    }

    /// Releases this service's sandbox container according to the card's
    /// `launcher_config.on_stop`.
    ///
    /// A no-op for a service that was not built through [`Self::sandbox_with_lifecycle`]:
    /// nothing was launched for it, so there is no container to release.
    ///
    /// # Errors
    ///
    /// See [`SandboxLifecycle::release`]: 199008 when the launcher fails and
    /// `ErrorKind::Unsupported` when the recorded launcher type is no longer registered.
    pub async fn release_sandbox(&self) -> Result<()> {
        let Some(binding) = self.sandbox.as_ref() else {
            tracing::debug!(
                card = %self.card.id,
                "release_sandbox is a no-op for a service without a lifecycle binding"
            );
            return Ok(());
        };
        binding
            .lifecycle
            .release(&binding.isolation_key, self.on_stop())
            .await
    }

    /// Stops every container that has been idle for at least this card's
    /// `launcher_config.idle_ttl_seconds`, returning the evicted isolation keys.
    ///
    /// Call it periodically from a host timer. Returns an empty list in local mode, when the
    /// launcher sets no `idle_ttl_seconds`, or when nothing is idle. Deletion ignores
    /// `on_stop`: an idle container is always destroyed (Python `_evict_idle` semantics).
    pub async fn evict_idle_sandboxes(&self) -> Vec<String> {
        let now_epoch_secs = self.sandbox_lifecycle().now();
        self.evict_idle_sandboxes_at(now_epoch_secs).await
    }

    /// [`Self::evict_idle_sandboxes`] with an explicit clock reading, for deterministic
    /// hosts and tests.
    pub async fn evict_idle_sandboxes_at(&self, now_epoch_secs: u64) -> Vec<String> {
        let Some(gateway) = self.card.gateway_config.as_ref() else {
            return Vec::new();
        };
        self.sandbox_lifecycle()
            .evict_idle_for(gateway, now_epoch_secs)
            .await
    }

    /// The lifecycle this service releases and evicts through.
    fn sandbox_lifecycle(&self) -> &SandboxLifecycle {
        match &self.sandbox {
            Some(binding) => &binding.lifecycle,
            None => crate::sandbox::lifecycle::global(),
        }
    }

    /// The card's `on_stop`, defaulting to Python's `delete`.
    fn on_stop(&self) -> OnStop {
        self.card
            .gateway_config
            .as_ref()
            .and_then(|gateway| gateway.launcher_config.as_ref())
            .map_or_else(OnStop::default, |launcher| launcher.on_stop)
    }

    /// Tool specs exposed by this service, with ids `"<card>.<kind>.<method>"`.
    ///
    /// Python parity: `SysOperationToolAdapter.extract_tools` renders one tool per
    /// registered operation; the filesystem surface contributes exactly ten, the shell
    /// surface two, and the code surface one.
    #[must_use]
    pub fn tool_specs(&self) -> Vec<OperationToolSpec> {
        let mut specs = fs_tool_specs(&self.card.id);
        specs.extend(shell_tool_specs(&self.card.id));
        specs.extend(crate::tool::code_tool_specs(&self.card.id));
        specs
    }

    /// Tool specs for one operation kind.
    #[must_use]
    pub fn tool_specs_for(&self, kind: OperationKind) -> Vec<OperationToolSpec> {
        match kind {
            OperationKind::Fs => fs_tool_specs(&self.card.id),
            OperationKind::Shell => shell_tool_specs(&self.card.id),
            OperationKind::Code => crate::tool::code_tool_specs(&self.card.id),
        }
    }

    /// Reads a file, reporting "no filesystem backend" as an error result.
    pub async fn read_file(
        &self,
        params: ReadFileParams,
    ) -> crate::result::OpResult<crate::fs::types::ReadFileData> {
        match &self.operations.fs {
            Some(fs) => fs.read_file(params).await,
            None => crate::result::fs_failure("read_file", "no filesystem backend"),
        }
    }

    /// Runs a snippet, reporting "no code backend" as an error result.
    pub async fn execute_code(
        &self,
        params: CodeExecParams,
    ) -> crate::result::OpResult<crate::code::types::CodeOutput> {
        match &self.operations.code {
            Some(code) => code.execute_code(params).await,
            None => crate::result::code_failure("execute_code", "no code backend"),
        }
    }
}

/// Convenience: build a local service and return its error message as a string.
///
/// # Errors
///
/// Propagates [`SysOperation::local`] failures.
pub fn try_local(config: crate::config::LocalWorkConfig) -> Result<SysOperation> {
    SysOperation::local(config)
}
