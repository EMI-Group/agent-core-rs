//! Filesystem operations: parameter types, the local backend, and the extension ops.
//!
//! [`local_fs`] is the seam the [`crate::SysOperation`] facade uses to obtain a local
//! filesystem backend; the sandbox backend is built by
//! [`crate::sandbox::operation::sandbox_operations`].
//!
//! ```
//! use openjiuwen_sysop::fs::types::ReadFileParams;
//!
//! let params = ReadFileParams::text("notes.txt").with_head(20);
//! assert_eq!(params.head, Some(20));
//! ```

use std::sync::Arc;

use openjiuwen_common::error::Result;

use crate::config::LocalWorkConfig;
use crate::operation::FsOperation;

pub mod ext;
pub mod list;
pub mod local;
pub mod lock;
mod read;
pub mod search;
pub mod slice;
mod transfer;
pub mod types;
mod write;

/// Builds the local filesystem backend for `config`.
///
/// # Errors
///
/// Returns `ErrorKind::Validation` when the configuration is invalid.
#[allow(clippy::needless_pass_by_value)] // the backend takes ownership of its configuration
pub fn local_fs(config: LocalWorkConfig) -> Result<Arc<dyn FsOperation>> {
    let backend = local::LocalFsOperation::new(config)?;
    Ok(Arc::new(backend))
}
