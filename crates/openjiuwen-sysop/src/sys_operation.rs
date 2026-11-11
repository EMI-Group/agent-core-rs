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
use crate::tool::{fs_tool_specs, shell_tool_specs, OperationToolSpec};

/// A configured sys-operation service.
#[derive(Clone)]
pub struct SysOperation {
    card: SysOperationCard,
    mode: OperationMode,
    operations: OperationSet,
    isolation_key_template: Option<String>,
}

impl std::fmt::Debug for SysOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SysOperation")
            .field("id", &self.card.id)
            .field("mode", &self.mode)
            .field("operations", &self.operations)
            .field("isolation_key_template", &self.isolation_key_template)
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
