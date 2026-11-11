//! Shell command execution: types, allowlist/denylist filtering, and the local backend.
//!
//! ```
//! use openjiuwen_sysop::shell::types::ShellExecParams;
//!
//! let params = ShellExecParams::new("echo hi").with_timeout_secs(5);
//! assert_eq!(params.effective_timeout_secs(300), 5);
//! ```

use std::sync::Arc;

use openjiuwen_common::error::Result;

use crate::config::LocalWorkConfig;
use crate::operation::ShellOperation;

pub mod allowlist;
pub mod local;
pub mod plan;
pub(crate) mod process;
pub mod registry;
pub mod types;

pub use local::LocalShellOperation;

/// Builds the local shell backend for `config`.
///
/// # Errors
///
/// Returns `ErrorKind::Validation` when the configuration is invalid.
#[allow(clippy::needless_pass_by_value)] // the backend takes ownership of its configuration
pub fn local_shell(config: LocalWorkConfig) -> Result<Arc<dyn ShellOperation>> {
    Ok(Arc::new(LocalShellOperation::new(config)?))
}
