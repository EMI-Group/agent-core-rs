//! Deferred paper-optimisation surface.

use serde_json::Value;

use openjiuwen_common::error::Result;

use super::not_implemented;

/// The surface name used by [`PaperOptimizer`]'s errors.
const SURFACE: &str = "paper optimiser";

/// Paper-writing optimiser — **deferred**.
///
/// Mirrors Python `openjiuwen.rsi.artifact_rsi.paper_opt.auto_research`, which
/// runs six `DeepAgent` stage agents over a paper repository and verifies the
/// result with a `LaTeX`/pyright/Pillow toolchain. Both the agent orchestration
/// and the host toolchain are out of scope for the offline port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaperOptimizer;

impl PaperOptimizer {
    /// Mirrors the optimiser's constructor; always reports that it is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`paper optimiser`).
    pub fn new() -> Result<Self> {
        Err(not_implemented(
            SURFACE,
            "needs six `DeepAgent` stage agents plus a Python host toolchain \
             (`LaTeX`, pyright, Pillow)",
        ))
    }

    /// Mirrors the auto-research pipeline's entry point; always reports that it
    /// is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`paper optimiser`).
    pub fn optimize(_request: Value) -> Result<Value> {
        Err(not_implemented(
            SURFACE,
            "needs six `DeepAgent` stage agents plus a Python host toolchain \
             (`LaTeX`, pyright, Pillow)",
        ))
    }
}
