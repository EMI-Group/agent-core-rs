//! Shell command execution: types, allowlist/denylist filtering, and the local backend.
//!
//! ```
//! use openjiuwen_sysop::shell::types::ShellExecParams;
//!
//! let params = ShellExecParams::new("echo hi").with_timeout_secs(5);
//! assert_eq!(params.effective_timeout_secs(300), 5);
//! ```

use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};

use crate::config::LocalWorkConfig;
use crate::operation::ShellOperation;

pub mod types;

/// Builds the local shell backend for `config`.
///
/// # Errors
///
/// Returns `ErrorKind::Validation` when the configuration is invalid.
#[allow(clippy::needless_pass_by_value)] // the backend takes ownership of its configuration
pub fn local_shell(config: LocalWorkConfig) -> Result<Arc<dyn ShellOperation>> {
    config.validate()?;
    Err(Error::unsupported(
        "the local shell backend is not implemented in this build",
    ))
}
