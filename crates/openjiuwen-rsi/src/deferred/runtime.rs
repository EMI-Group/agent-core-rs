//! Deferred runtime-extension loading.
//!
//! Subprocess execution, git operations, CI gates, the Python toolchain, and
//! the subprocess program evaluator are **implemented** in [`crate::runtime`].
//! Only the extension loader stays here: it hot-loads Python modules, and
//! Python's import machinery has no Rust analogue.

use serde_json::Value;

use openjiuwen_common::error::Result;

use super::not_implemented;

/// The surface name used by [`RuntimeExtensionLoader`]'s errors.
const SURFACE_RUNTIME_EXTENSION_LOADER: &str = "runtime extension loader";

/// Loader for Python runtime extensions — **deferred**.
///
/// Mirrors Python's `harness_rsi.auto_harness.infra.runtime_extension_loader`,
/// which hot-loads extension modules by synthesising `sys.modules` entries and
/// running static checks. Python import machinery has no Rust analogue here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeExtensionLoader;

impl RuntimeExtensionLoader {
    /// Mirrors the loader's constructor; always reports that it is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`runtime extension loader`).
    pub fn new() -> Result<Self> {
        Err(not_implemented(
            SURFACE_RUNTIME_EXTENSION_LOADER,
            "needs synthetic `sys.modules` hot-loading of Python runtime extensions",
        ))
    }

    /// Mirrors `RuntimeExtensionLoader.load`; always reports that it is
    /// deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`runtime extension loader`).
    pub fn load(_request: Value) -> Result<Value> {
        Err(not_implemented(
            SURFACE_RUNTIME_EXTENSION_LOADER,
            "needs synthetic `sys.modules` hot-loading of Python runtime extensions",
        ))
    }
}
