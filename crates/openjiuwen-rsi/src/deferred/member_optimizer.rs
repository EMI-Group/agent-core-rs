//! Deferred member-optimisation surface.

use serde_json::Value;

use openjiuwen_common::error::Result;

use super::not_implemented;

/// The surface name used by [`MemberOptimizer`]'s errors.
const SURFACE: &str = "member optimiser";

/// Team-member optimisation loop — **deferred**.
///
/// Mirrors Python's
/// `harness_rsi.{evaluation_result_analyzer,member_optimizer,improver_evolution}`,
/// which diagnose a member's failures, attribute them to a cause, and plan an
/// improvement through LLM prompts. There is no portable core to port, so this
/// is an API-shaped stub.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemberOptimizer;

impl MemberOptimizer {
    /// Mirrors the optimiser's constructor; always reports that it is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`member optimiser`).
    pub fn new() -> Result<Self> {
        Err(not_implemented(
            SURFACE,
            "needs agent-driven diagnosis, attribution, and action planning over LLM prompts",
        ))
    }

    /// Mirrors the member-optimisation entry point; always reports that it is
    /// deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`member optimiser`).
    pub fn optimize(_request: Value) -> Result<Value> {
        Err(not_implemented(
            SURFACE,
            "needs agent-driven diagnosis, attribution, and action planning over LLM prompts",
        ))
    }
}
