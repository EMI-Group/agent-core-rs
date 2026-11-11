//! Local and sandbox filesystem, shell, and code operations.
//!
//! This crate maps to the Python package `openjiuwen.core.sys_operation`: a
//! [`SysOperation`] facade built from a [`SysOperationCard`], exposing filesystem
//! ([`FsOperation`]), shell ([`ShellOperation`]), and code ([`CodeOperation`]) backends.
//! The local backend runs everything in-process; the sandbox backend forwards the same
//! calls to a provider registered through [`sandbox::provider`].
//!
//! Two Python behaviors are preserved deliberately and are easy to get wrong:
//!
//! - Operations report failures as values. Every call returns [`OpResult`] with Python's
//!   status code (`199_003` filesystem, `199_004` shell, `199_005` code, `199_011` write
//!   conflict), so a caller can port Python error handling without translating
//!   exceptions. [`OpResult::into_result`] converts to an idiomatic `Result`.
//! - `write_file` prepends a newline by default (`prepend_newline = true`). Use
//!   [`fs::types::WriteFileParams::raw`] when the file must contain exactly the content.
//!
//! ```
//! use openjiuwen_sysop::card::SysOperationCard;
//! use openjiuwen_sysop::mode::OperationMode;
//! use openjiuwen_sysop::status::SysOpStatus;
//!
//! let card = SysOperationCard::new("sys_op").with_mode(OperationMode::Local);
//! assert_eq!(card.id, "sys_op");
//! assert_eq!(SysOpStatus::FsExecutionError.code(), 199_003);
//! ```
//!
//! # Security model
//!
//! - Every path is resolved through [`workspace::PathPolicy`]: lexical `..` removal,
//!   symlink resolution for the existing prefix, and — when `restrict_to_sandbox` is set —
//!   containment under the configured roots.
//! - Shell commands pass a first-token allowlist and an optional regex deny list before a
//!   child process is spawned; [`ShellOperation::execute_argv`] avoids the shell entirely.
//! - Output capture is bounded by `LocalWorkConfig::max_output_bytes` with an explicit
//!   truncation marker, and child environments are scoped by `LocalWorkConfig::env_policy`.
//! - There is no `unsafe` code and no panic on malformed input: every failure path returns
//!   an [`OpResult`] or an [`openjiuwen_common::error::Error`].

#![forbid(unsafe_code)]
#![allow(clippy::module_name_repetitions)]

pub mod card;
pub mod code;
pub mod config;
pub mod constants;
pub mod encoding;
pub mod env;
pub mod fs;
pub mod mode;
pub mod operation;
pub mod result;
pub mod sandbox;
pub mod shell;
pub mod status;
pub mod sys_operation;
pub mod tool;
pub mod util;
pub mod workspace;

pub use card::SysOperationCard;
pub use config::{BinaryPolicy, EnvInheritMode, EnvPolicy, LocalWorkConfig};
pub use mode::{OperationKind, OperationMode};
pub use operation::{
    CodeOperation, FsExtOperation, FsOperation, OpStream, OperationSet, ShellOperation,
};
pub use result::{code_failure, fs_failure, shell_failure, OpResult};
pub use status::SysOpStatus;
pub use sys_operation::SysOperation;

/// The crate name, useful in diagnostics and documentation.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
