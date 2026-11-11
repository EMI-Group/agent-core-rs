//! Code execution: parameter types and the local subprocess backend.
//!
//! ```
//! use openjiuwen_sysop::code::types::{CodeExecParams, Language};
//!
//! let params = CodeExecParams::new("print(1)");
//! assert_eq!(params.language, Language::Python);
//! ```

use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};

use crate::config::LocalWorkConfig;
use crate::operation::CodeOperation;

pub mod types;

/// Builds the local code backend for `config`.
///
/// # Errors
///
/// Returns `ErrorKind::Validation` when the configuration is invalid.
#[allow(clippy::needless_pass_by_value)] // the backend takes ownership of its configuration
pub fn local_code(config: LocalWorkConfig) -> Result<Arc<dyn CodeOperation>> {
    config.validate()?;
    Err(Error::unsupported(
        "the local code backend is not implemented in this build",
    ))
}
